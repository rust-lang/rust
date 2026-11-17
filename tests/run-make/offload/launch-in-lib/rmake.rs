//@ needs-offload

// Test launching kernels in a different "crate", both generic and non-generic

use run_make_support::symbols::object_contains_any_symbol_substring;
use run_make_support::{rfs, rustc};

fn main() {
    rfs::create_dir("host");
    rfs::create_dir("host-metadata");
    rfs::create_dir("device");

    rustc()
        .input("launch.rs")
        .arg("-Zunstable-options")
        .arg("-Zoffload=HostMetadata=launch.manifest")
        .arg("-Csymbol-mangling-version=v0")
        .out_dir("host-metadata")
        .run();

    rustc()
        .input("launch.rs")
        .cfg("device")
        .arg("-Zunstable-options")
        .arg("-Zoffload=Device=launch.manifest")
        .arg("-Csymbol-mangling-version=v0")
        .arg("-Ccodegen-units=1")
        .emit("link,obj")
        .out_dir("device")
        .run();

    rustc()
        .input("launch.rs")
        .arg("-Zunstable-options")
        .arg("-Zoffload=Host")
        .arg("-Csymbol-mangling-version=v0")
        .out_dir("host")
        .run();

    rustc()
        .input("main.rs")
        .extern_("launch", "host/liblaunch.rlib")
        .arg("-Zunstable-options")
        .arg("-Zoffload=HostMetadata=main.manifest")
        .arg("-Csymbol-mangling-version=v0")
        .arg("-Clto=fat")
        .run();

    rustc()
        .input("main.rs")
        .cfg("device")
        .extern_("launch", "device/liblaunch.rlib")
        .arg("-Zunstable-options")
        .arg("-Zoffload=Device=main.manifest")
        .arg("-Csymbol-mangling-version=v0")
        .arg("-Clto=fat")
        .emit("obj")
        .run();

    assert!(object_contains_any_symbol_substring(
        "device/launch.o",
        &["_RINvC6launch14generic_kerneldEB2_"]
    ));
    assert!(object_contains_any_symbol_substring(
        "device/launch.o",
        &["_RINvC6launch14generic_kernelfEB2_"]
    ));
    assert!(object_contains_any_symbol_substring("device/launch.o", &["_RNvC6launch6kernel"]));
}
