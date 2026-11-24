//@ needs-offload

// Test defining kernels in a different "crate", both generic and non-generic

use run_make_support::symbols::object_contains_any_symbol_substring;
use run_make_support::{rfs, rustc};

fn main() {
    rfs::create_dir("host");
    rfs::create_dir("device");

    rustc().input("kernels.rs").arg("-Csymbol-mangling-version=v0").out_dir("host").run();

    rustc()
        .input("kernels.rs")
        .arg("-Zunstable-options")
        .arg("-Zoffload=Host")
        .arg("-Csymbol-mangling-version=v0")
        .emit("offload-info")
        .out_dir("host")
        .run();

    rustc()
        .input("kernels.rs")
        .cfg("device")
        .arg("-Zunstable-options")
        .arg("-Zoffload=Device=host/kernels.roffload")
        .arg("-Csymbol-mangling-version=v0")
        .arg("-Ccodegen-units=1")
        .emit("link,obj")
        .out_dir("device")
        .run();

    rustc()
        .input("main.rs")
        .extern_("kernels", "host/libkernels.rlib")
        .arg("-Zunstable-options")
        .arg("-Zoffload=Host")
        .arg("-Csymbol-mangling-version=v0")
        .arg("-Clto=fat")
        .emit("offload-info")
        .out_dir("host")
        .run();

    rustc()
        .input("main.rs")
        .cfg("device")
        .extern_("kernels", "device/libkernels.rlib")
        .arg("-Zunstable-options")
        .arg("-Zoffload=Device=host/main.roffload")
        .arg("-Csymbol-mangling-version=v0")
        .arg("-Clto=fat")
        .emit("obj")
        .out_dir("host")
        .run();

    // The generic kernels are instantiated in the device compilation of the calling crate
    assert!(object_contains_any_symbol_substring("host/main.o", &["7kernels14generic_kernelfE"]));
    assert!(object_contains_any_symbol_substring("host/main.o", &["7kernels14generic_kernellE"]));

    // The non-generic `kernels::kernel` is generated once by the crate that defines it.
    assert!(object_contains_any_symbol_substring("device/kernels.o", &["7kernels6kernel"]));
    assert!(!object_contains_any_symbol_substring("host/main.o", &["7kernels6kernel"]));
}
