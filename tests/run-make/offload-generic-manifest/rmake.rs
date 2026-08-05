// Tests the offload manifest pipeline for generic kernels

use run_make_support::rustc;
use run_make_support::symbols::object_contains_any_symbol_substring;

fn main() {
    rustc()
        .input("generic.rs")
        .arg("-Zunstable-options")
        .arg("-Zoffload=HostMetadata=generic.manifest")
        .arg("-Clto=fat")
        .emit("metadata")
        .run();

    rustc()
        .input("generic.rs")
        .cfg("device")
        .arg("-Zunstable-options")
        .arg("-Zoffload=Device=generic.manifest")
        .arg("-Clto=fat")
        .emit("obj")
        .run();

    assert!(object_contains_any_symbol_substring("generic.o", &["6kernelfEB2_"]));
    assert!(object_contains_any_symbol_substring("generic.o", &["6kernellEB2_"]));

    rustc()
        .input("generic.rs")
        .cfg("device")
        .arg("-Zunstable-options")
        .arg("-Zoffload=Device")
        .arg("-Clto=fat")
        .emit("obj")
        .run();

    assert!(!object_contains_any_symbol_substring("generic.o", &["6kernel"]));
}
