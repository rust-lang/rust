//! This test builds and runs a basic UEFI application on QEMU for various targets.
//!
//! You must have the relevant OVMF or AAVMF firmware installed for this to work.
//!
//! Requires: qemu-system-x86_64, qemu-system-aarch64
//!           OVMF/AAVMF firmware
//!
//! Note: test assumes `/uefi_qemu_test` exists and is a self-contained crate.

//@ only-uefi

use std::path::Path;

use run_make_support::{cargo, cmd, path, rfs};

fn main() {
    let target = run_make_support::target();

    let (boot_filename, ovmf_dir, ovmf_code_name, ovmf_vars_name, qemu, machine, cpu) =
        match target.as_str() {
            "aarch64-unknown-uefi" => (
                "bootaa64.efi",
                Path::new("/usr/share/AAVMF"),
                "AAVMF_CODE.fd",
                "AAVMF_VARS.fd",
                "qemu-system-aarch64",
                "virt",
                "cortex-a72",
            ),
            "i686-unknown-uefi" => (
                "bootia32.efi",
                Path::new("/usr/share/OVMF"),
                "OVMF32_CODE_4M.secboot.fd",
                "OVMF32_VARS_4M.fd",
                "qemu-system-x86_64",
                "q35",
                "qemu64",
            ),
            "x86_64-unknown-uefi" => (
                "bootx64.efi",
                Path::new("/usr/share/OVMF"),
                "OVMF_CODE_4M.fd",
                "OVMF_VARS_4M.fd",
                "qemu-system-x86_64",
                "q35",
                "qemu64",
            ),
            _ => panic!("unsupported target {target}"),
        };

    let tmp = std::env::temp_dir();
    let test_crate = tmp.join("uefi_qemu_test");
    rfs::copy_dir_all(path("uefi_qemu_test"), &test_crate);

    cargo().args(&["build", "--target", &target]).current_dir(&test_crate).run();

    // Prepare ESP
    let esp = test_crate.join("esp");
    let boot = esp.join("efi/boot");
    rfs::create_dir_all(&boot);

    let src_efi = test_crate.join("target").join(&target).join("debug/uefi_qemu_test.efi");
    let dst_efi = boot.join(boot_filename);
    rfs::copy(&src_efi, &dst_efi);

    // Copy OVMF files
    let code = ovmf_dir.join(ovmf_code_name);
    let vars_src = ovmf_dir.join(ovmf_vars_name);
    let vars_dst = tmp.join("vars.fd");
    rfs::copy(&vars_src, &vars_dst);

    let output = cmd(qemu)
        .args(["-machine", machine])
        .args(["-cpu", cpu])
        .args(["-display", "none"])
        .args(["-serial", "stdio"])
        .args(["-drive", &format!("if=pflash,format=raw,readonly=on,file={}", code.display())])
        .args(["-drive", &format!("if=pflash,format=raw,readonly=off,file={}", vars_dst.display())])
        .args(["-drive", &format!("format=raw,file=fat:rw:{}", esp.display())])
        .run()
        .stdout_utf8();

    assert!(output.contains("Hello World!"), "invalid output for {target}:\n{output}");
}
