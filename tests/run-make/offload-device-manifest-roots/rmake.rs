// With a manifest, the device pass compiles exactly the kernels the host launches and whatever
// they reach, nothing else. That includes reaching into dependencies: their functions are made
// available for instantiation by `-Zoffload=Device` itself, so the dependencies do not have to
// codegen anything for the device, and the kernel crate ends up self-contained.

//@ needs-offload

use std::path::Path;

use run_make_support::{rfs, rustc};

fn defines(ll: &str, name: &str) -> bool {
    ll.lines().any(|line| line.starts_with("define") && line.contains(name))
}

fn main() {
    // Host build of the dependency, so the metadata pass can resolve `dep::helper`.
    rustc().input("dep.rs").run();

    // Pass 1: Our manifest now includes both generic and non-generic kernels that were launched.
    rustc()
        .input("host.rs")
        .extern_("dep", "libdep.rlib")
        .arg("-Zunstable-options")
        .arg("-Zoffload=HostMetadata=kernels.manifest")
        .arg("-Csymbol-mangling-version=v0")
        .arg("-Clto=fat")
        .emit("metadata")
        .run();

    // Pass 2a: the dependency for the device. It reads a manifest naming kernels in a crate it
    // has never heard of, has no launched kernels of its own, and so has nothing to codegen.
    // FIXME(offload): In the future, we should add better errorhandling here. It's fine to not find
    // the kernels mentioned in the manifest if this is just a dep. However, if the Manifest entry
    // names this crate and the path to the Kernel does not resolve, then we should error. The
    // decoder should be able to tell the difference between both cases.
    rustc()
        .input("dep.rs")
        .arg("-Zunstable-options")
        .arg("-Zoffload=Device=kernels.manifest")
        .arg("-Csymbol-mangling-version=v0")
        .codegen_units(1)
        .emit("link,llvm-ir")
        .out_dir("device")
        .run();
    if Path::new("device/dep.ll").exists() {
        let dep_ll = rfs::read_to_string("device/dep.ll");
        assert!(!defines(&dep_ll, "helper"), "`dep::helper` was codegened in its own crate");
    }

    // Pass 2b: the kernel crate for the device.
    rustc()
        .input("host.rs")
        .extern_("dep", "device/libdep.rlib")
        .arg("-Zunstable-options")
        .arg("-Zoffload=Device=kernels.manifest")
        .arg("-Csymbol-mangling-version=v0")
        .arg("-Clto=fat")
        .codegen_units(1)
        .emit("llvm-ir")
        .out_dir("device")
        .run();
    let ll = rfs::read_to_string("device/host.ll");
    assert!(defines(&ll, "launched"), "the launched kernel is missing");
    assert!(defines(&ll, "helper"), "`dep::helper` was not instantiated in the kernel crate");
    assert!(!defines(&ll, "dormant"), "an unlaunched kernel was compiled");
    assert!(!defines(&ll, "plain_pub"), "a function no kernel reaches was compiled");
    // The entry point would be a root under the usual rules; on the device it is not launched.
    assert!(!defines(&ll, "main"), "the host entry point was compiled for the device");
}
