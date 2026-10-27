"""Download the official DDL-I test archive and retain only fake images and masks."""

import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import time

import requests
from modelscope.hub.api import HubApi


DATASET = "DDL_dataset"
NAMESPACE = "DDLteam"
REMOTE_FILE = "phase2/track1/test.tar.gz"
CHUNK_SIZE = 8 * 1024 * 1024


def log(message):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), message, flush=True)


def expected_file(api):
    tree = api.list_repo_tree(DATASET, NAMESPACE, "master", "phase2/track1", recursive=False)
    for entry in tree["Data"]["Files"]:
        if entry["Path"] == REMOTE_FILE:
            return int(entry["Size"]), entry["Sha256"].lower()
    raise RuntimeError(f"Official file not found: {REMOTE_FILE}")


def download(api, output_dir, expected_size):
    archive = output_dir / "DDL-I-test.tar.gz"
    partial = output_dir / "DDL-I-test.tar.gz.part"
    if archive.exists():
        if archive.stat().st_size != expected_size:
            raise RuntimeError("Existing archive has an unexpected size")
        return archive

    url = api.get_dataset_file_url(REMOTE_FILE, DATASET, NAMESPACE)
    last_log = 0
    while True:
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == expected_size:
            break
        if offset > expected_size:
            raise RuntimeError("Partial archive is larger than the official file")
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(30, 180)) as response:
                response.raise_for_status()
                if offset:
                    expected_range = f"bytes {offset}-"
                    if response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(expected_range):
                        raise RuntimeError("Server did not honor the resume offset")
                with partial.open("ab") as handle:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        now = time.monotonic()
                        if now - last_log >= 30:
                            size = handle.tell()
                            log(f"Downloaded {size / 1e9:.2f}/{expected_size / 1e9:.2f} GB ({size / expected_size:.1%})")
                            last_log = now
        except requests.RequestException as error:
            log(f"Transfer interrupted: {error}; resuming in 10 seconds")
            time.sleep(10)

    if partial.stat().st_size != expected_size:
        raise RuntimeError("Download ended before the official file size was reached")
    os.replace(partial, archive)
    return archive


def verify_sha256(archive, expected_digest):
    log("Verifying official SHA-256")
    digest = hashlib.sha256()
    with archive.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    if digest.hexdigest().lower() != expected_digest:
        raise RuntimeError("SHA-256 mismatch; archive retained for diagnosis")
    log("SHA-256 verified")


def copy_member(source, destination, expected_size):
    if destination.exists():
        if destination.stat().st_size != expected_size:
            raise RuntimeError(f"Existing output has a different size: {destination}")
        return
    temporary = destination.with_name(destination.name + ".part")
    with temporary.open("wb") as handle:
        shutil.copyfileobj(source, handle, length=CHUNK_SIZE)
    if temporary.stat().st_size != expected_size:
        raise RuntimeError(f"Extracted size mismatch: {destination}")
    os.replace(temporary, destination)


def extract_fake_pairs(archive, output_dir):
    images_dir = output_dir / "Tp"
    masks_dir = output_dir / "Gt"
    images_dir.mkdir(exist_ok=True)
    masks_dir.mkdir(exist_ok=True)
    fake_names = None
    image_count = 0
    mask_count = 0
    seen = set()
    with tarfile.open(archive, mode="r|gz") as members:
        for member in members:
            if not member.isfile():
                continue
            parts = PurePosixPath(member.name).parts
            name = parts[-1]
            if name == "labels.txt":
                content = members.extractfile(member).read().decode("utf-8")
                fake_names = {line.rsplit(",", 1)[0] for line in content.splitlines() if line.rsplit(",", 1)[-1].strip() == "1"}
                log(f"Official test labels: {len(content.splitlines())} total, {len(fake_names)} fake")
                continue
            if fake_names is None:
                raise RuntimeError("The test archive did not provide labels before images")
            if name not in fake_names:
                continue
            kind = "Gt" if "masks" in parts else "Tp"
            if kind == "Tp" and "masks" in parts:
                raise RuntimeError(f"Cannot classify archive member: {member.name}")
            key = (kind, name)
            if key in seen:
                raise RuntimeError(f"Duplicate archive member: {member.name}")
            seen.add(key)
            destination = (masks_dir if kind == "Gt" else images_dir) / name
            copy_member(members.extractfile(member), destination, member.size)
            if kind == "Gt":
                mask_count += 1
            else:
                image_count += 1
            if (image_count + mask_count) % 10000 == 0:
                log(f"Extracted {image_count} fake images and {mask_count} masks")

    if fake_names is None or image_count != len(fake_names) or mask_count != len(fake_names):
        raise RuntimeError(f"Incomplete pairs: expected {len(fake_names) if fake_names else 0}; got {image_count} images and {mask_count} masks")
    log(f"Complete: {image_count} fake images and {mask_count} masks")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    api = HubApi()
    expected_size, expected_digest = expected_file(api)
    free = shutil.disk_usage(args.output_dir).free
    if free < expected_size * 2:
        raise RuntimeError(f"Insufficient free space: {free} bytes for {expected_size}-byte archive and extraction")
    log(f"Official DDL-I test archive: {expected_size} bytes")
    archive = download(api, args.output_dir, expected_size)
    verify_sha256(archive, expected_digest)
    extract_fake_pairs(archive, args.output_dir)
    archive.unlink()
    log("Removed temporary official archive; only the selected image and GT files remain")


if __name__ == "__main__":
    main()
