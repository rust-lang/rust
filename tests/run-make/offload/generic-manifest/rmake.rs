//@ needs-offload

// Tests the offload manifest pipeline for generic kernels

use run_make_support::rustc;
use run_make_support::symbols::object_contains_any_symbol_substring;

fn main() {
    rustc()
        .input("generic.rs")
        .arg("-Zunstable-options")
        .arg("-Zoffload=Host")
        .arg("-Csymbol-mangling-version=v0")
        .arg("-Clto=fat")
        .emit("offload-info=generic.roffload")
        .run();

    rustc()
        .input("generic.rs")
        .cfg("device")
        .arg("-Zunstable-options")
        .arg("-Zoffload=Device=generic.roffload")
        .arg("-Csymbol-mangling-version=v0")
        .arg("-Clto=fat")
        .emit("obj")
        .run();

    assert!(object_contains_any_symbol_substring("generic.o", &["6kernelfEB2_"]));
    assert!(object_contains_any_symbol_substring("generic.o", &["6kernellEB2_"]));

    let p = rustc()
        .input("generic.rs")
        .cfg("device")
        .arg("-Zunstable-options")
        .arg("-Zoffload=Device")
        .arg("-Csymbol-mangling-version=v0")
        .arg("-Clto=fat")
        .emit("obj")
        .run_fail();
    assert!(p.stderr_utf8().contains("generic offload kernel"));
    assert!(p.stderr_utf8().contains("is not instantiated"));
}
