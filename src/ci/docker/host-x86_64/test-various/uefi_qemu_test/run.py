#!/usr/bin/env python3

import os
import shutil
import subprocess
import sys
import tempfile

from pathlib import Path


def run(*cmd, capture=False, check=True, env=None):
    """Print and run a command, optionally capturing the output."""
    cmd = [str(p) for p in cmd]
    print(' '.join(cmd))
    return subprocess.run(cmd,
                          capture_output=capture,
                          check=check,
                          env=env,
                          text=True)


def build_and_run(tmp_dir):
    host_artifacts = Path('/checkout/obj/build/x86_64-unknown-linux-gnu')
    stage0 = host_artifacts / 'stage0/bin'
    stage2 = host_artifacts / 'stage2/bin'

    env = dict(os.environ)
    env['PATH'] = '{}:{}:{}'.format(stage2, stage0, env['PATH'])

    # Copy the test create into `tmp_dir`.
    test_crate = Path(tmp_dir) / 'uefi_qemu_test'
    shutil.copytree('/uefi_qemu_test', test_crate)

    # Build the UEFI executable.
    target = 'x86_64-unknown-uefi'
    run('cargo',
        'build',
        '--manifest-path',
        test_crate / 'Cargo.toml',
        '--target',
        target,
        env=env)

    # Create a mock EFI System Partition in a subdirectory.
    esp = test_crate / 'esp'
    boot = esp / 'efi/boot'
    os.makedirs(boot, exist_ok=True)

    # Copy the executable into the ESP.
    src_exe_path = test_crate / 'target' / target / 'debug/uefi_qemu_test.efi'
    shutil.copy(src_exe_path, boot / 'bootx64.efi')

    # Run the executable in QEMU and capture the output.
    qemu = 'qemu-system-x86_64'
    ovmf_dir = Path('/usr/share/OVMF')
    ovmf_code = ovmf_dir / 'OVMF_CODE.fd'
    ovmf_vars = ovmf_dir / 'OVMF_VARS.fd'
    output = run(qemu,
                 '-display',
                 'none',
                 '-serial',
                 'stdio',
                 '-drive',
                 f'if=pflash,format=raw,readonly=on,file={ovmf_code}',
                 '-drive',
                 f'if=pflash,format=raw,readonly=on,file={ovmf_vars}',
                 '-drive',
                 f'format=raw,file=fat:rw:{esp}',
                 capture=True,
                 # Ubuntu 20.04 (which is what the Dockerfile currently
                 # uses) provides QEMU 4.2.1, which segfaults on
                 # shutdown under some circumstances. That has been
                 # fixed in newer versions of QEMU, but for now just
                 # don't check the exit status.
                 check=False).stdout

    if 'Hello World!' in output:
        print('VM produced expected output')
    else:
        print('unexpected VM output:')
        print('---start---')
        print(output)
        print('---end---')
        sys.exit(1)


def main():
    # Create a temporary directory so that we have a writeable
    # workspace.
    with tempfile.TemporaryDirectory() as tmp_dir:
        build_and_run(tmp_dir)


if __name__ == "__main__":
    main()
