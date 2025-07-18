//@ needs-target-std
//
// Generating metadata alongside remap-path-prefix would fail to actually remap the path
// in the metadata. After this was fixed in #85344, this test checks that "auxiliary" is being
// successfully remapped to "/the/aux" in the rmeta files.
// See https://github.com/rust-lang/rust/pull/85344

use run_make_support::bstr::ByteSlice;
use run_make_support::{bstr, is_darwin, rfs, rustc};

fn main() {
    let mut out_simple = rustc();
    let mut out_object = rustc();
    let mut out_macro = rustc();
    let mut out_diagobj = rustc();
    out_simple
        .remap_path_prefix("auxiliary", "/the/aux")
        .crate_type("lib")
        .emit("metadata")
        .input("auxiliary/lib.rs");
    out_object
        .remap_path_prefix("auxiliary", "/the/aux")
        .crate_type("lib")
        .emit("metadata")
        .input("auxiliary/lib.rs");
    out_macro
        .remap_path_prefix("auxiliary", "/the/aux")
        .crate_type("lib")
        .emit("metadata")
        .input("auxiliary/lib.rs");
    out_diagobj
        .remap_path_prefix("auxiliary", "/the/aux")
        .crate_type("lib")
        .emit("metadata")
        .input("auxiliary/lib.rs");

    out_simple.run();
    rmeta_contains("/the/aux/lib.rs");
    rmeta_not_contains("auxiliary");

    out_object.arg("-Zremap-path-scope=object");
    out_macro.arg("-Zremap-path-scope=macro");
    out_diagobj.arg("-Zremap-path-scope=diagnostics,object");
    if is_darwin() {
        out_object.arg("-Csplit-debuginfo=off");
        out_macro.arg("-Csplit-debuginfo=off");
        out_diagobj.arg("-Csplit-debuginfo=off");
    }

    out_object.run();
    rmeta_contains("/the/aux/lib.rs");
    rmeta_contains("auxiliary");
    out_macro.run();
    rmeta_contains("/the/aux/lib.rs");
    rmeta_contains("auxiliary");
    out_diagobj.run();
    rmeta_contains("/the/aux/lib.rs");
    rmeta_not_contains("auxiliary");
}

//FIXME(Oneirical): These could be generalized into run_make_support
// helper functions.
#[track_caller]
fn rmeta_contains(expected: &str) {
    // Normalize to account for path differences in Windows.
    if !bstr::BString::from(rfs::read("liblib.rmeta")).replace(b"\\", b"/").contains_str(expected) {
        eprintln!("=== FILE CONTENTS (LOSSY) ===");
        eprintln!("{}", String::from_utf8_lossy(&rfs::read("liblib.rmeta")));
        eprintln!("=== SPECIFIED TEXT ===");
        eprintln!("{}", expected);
        panic!("specified text was not found in file");
    }
}

#[track_caller]
fn rmeta_not_contains(expected: &str) {
    // Normalize to account for path differences in Windows.
    if bstr::BString::from(rfs::read("liblib.rmeta")).replace(b"\\", b"/").contains_str(expected) {
        eprintln!("=== FILE CONTENTS (LOSSY) ===");
        eprintln!("{}", String::from_utf8_lossy(&rfs::read("liblib.rmeta")));
        eprintln!("=== SPECIFIED TEXT ===");
        eprintln!("{}", expected);
        panic!("specified text was not found in file");
    }
}
