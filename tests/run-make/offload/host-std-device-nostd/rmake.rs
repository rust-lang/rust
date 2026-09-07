//@ needs-offload

// Tests offload with no-std in device and std in host

use run_make_support::{cwd, rfs, rustc};

fn main() {
    rustc()
        .input("example.rs")
        .arg("-Zunstable-options")
        .arg("-Zoffload=HostMetadata=example.manifest")
        .arg("-Csymbol-mangling-version=v0")
        .arg("-Clto=fat")
        .emit("metadata")
        .run();

    rustc()
        .input("example.rs")
        .cfg("device")
        .arg("-Zunstable-options")
        .arg("-Zoffload=Device=example.manifest")
        .arg("-Csymbol-mangling-version=v0")
        .arg("-Clto=fat")
        .arg("-Cpanic=abort")
        .emit("obj")
        .run();

    rfs::write(cwd().join("device.bin"), [0u8; 8]);

    rustc()
        .input("example.rs")
        .arg("-Zunstable-options")
        .arg(format!("-Zoffload=Host={}", cwd().join("device.bin").display()))
        .arg("-Csymbol-mangling-version=v0")
        .arg("-Clto=fat")
        .emit("obj")
        .run();

    assert!(cwd().join("host.o").exists());
}
