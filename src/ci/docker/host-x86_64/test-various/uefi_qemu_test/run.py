#!/usr/bin/env python3

import os
import shutil
import subprocess
import sys
import tempfile

from pathlib import Path

TARGET_AARCH64 = 'aarch64-unknown-uefi'
TARGET_I686 = 'i686-unknown-uefi'
TARGET_X86_64 = 'x86_64-unknown-uefi'

def run(*cmd, capture=False, check=True, env=None, timeout=None):
    """Print and run a command, optionally capturing the output."""
    cmd = [str(p) for p in cmd]
    print(' '.join(cmd))
    return subprocess.run(cmd,
                          capture_output=capture,
                          check=check,
                          env=env,
                          text=True,
                          timeout=timeout)

def build_and_run(tmp_dir, target):
    if target == TARGET_AARCH64:
        boot_file_name = 'bootaa64.efi'
        ovmf_dir = Path('/usr/share/AAVMF')
        ovmf_code = 'AAVMF_CODE.fd'
        ovmf_vars = 'AAVMF_VARS.fd'
        qemu = 'qemu-system-aarch64'
        machine = 'virt'
        cpu = 'cortex-a72'
    elif target == TARGET_I686:
        boot_file_name = 'bootia32.efi'
        ovmf_dir = Path('/usr/share/OVMF')
        ovmf_code = 'OVMF32_CODE_4M.secboot.fd'
        ovmf_vars = 'OVMF32_VARS_4M.fd'
        # The i686 target intentionally uses 64-bit qemu; the important
        # difference is that the OVMF code provides a 32-bit environment.
        qemu = 'qemu-system-x86_64'
        machine = 'q35'
        cpu = 'qemu64'
    elif target == TARGET_X86_64:
        boot_file_name = 'bootx64.efi'
        ovmf_dir = Path('/usr/share/OVMF')
        ovmf_code = 'OVMF_CODE.fd'
        ovmf_vars = 'OVMF_VARS.fd'
        qemu = 'qemu-system-x86_64'
        machine = 'q35'
        cpu = 'qemu64'
    else:
        raise KeyError('invalid target')

    host_artifacts = Path('/checkout/obj/build/x86_64-unknown-linux-gnu')
    stage0 = host_artifacts / 'stage0/bin'
    stage2 = host_artifacts / 'stage2/bin'

    env = dict(os.environ)
    env['PATH'] = '{}:{}:{}'.format(stage2, stage0, env['PATH'])

    # Copy the test create into `tmp_dir`.
    test_crate = Path(tmp_dir) / 'uefi_qemu_test'
    shutil.copytree('/uefi_qemu_test', test_crate)

    # Build the UEFI executable.
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
    shutil.copy(src_exe_path, boot / boot_file_name)
    print(src_exe_path, boot / boot_file_name)

    # Select the appropriate EDK2 build.
    ovmf_code = ovmf_dir / ovmf_code
    ovmf_vars = ovmf_dir / ovmf_vars

    # Make a writable copy of the vars file. aarch64 doesn't boot
    # correctly with read-only vars.
    ovmf_rw_vars = Path(tmp_dir) / 'vars.fd'
    shutil.copy(ovmf_vars, ovmf_rw_vars)

    # Run the executable in QEMU and capture the output.
    output = run(qemu,
                 '-machine',
                 machine,
                 '-cpu',
                 cpu,
                 '-display',
                 'none',
                 '-serial',
                 'stdio',
                 '-drive',
                 f'if=pflash,format=raw,readonly=on,file={ovmf_code}',
                 '-drive',
                 f'if=pflash,format=raw,readonly=off,file={ovmf_rw_vars}',
                 '-drive',
                 f'format=raw,file=fat:rw:{esp}',
                 capture=True,
                 # Ubuntu 20.04 (which is what the Dockerfile currently
                 # uses) provides QEMU 4.2.1, which segfaults on
                 # shutdown under some circumstances. That has been
                 # fixed in newer versions of QEMU, but for now just
                 # don't check the exit status.
                 check=False,
                 # Set a timeout to kill the VM in case something goes wrong.
                 timeout=60).stdout

    if 'Hello World!' in output:
        print('VM produced expected output')
    else:
        print('unexpected VM output:')
        print('---start---')
        print(output)
        print('---end---')
        sys.exit(1)


def main():
    targets = [TARGET_AARCH64, TARGET_I686, TARGET_X86_64]

    for target in targets:
        # Create a temporary directory so that we have a writeable
        # workspace.
        with tempfile.TemporaryDirectory() as tmp_dir:
            build_and_run(tmp_dir, target)


if __name__ == "__main__":
    main()
