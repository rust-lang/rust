// Generating metadata alongside remap-path-prefix would fail to actually remap the path
// in the metadata. After this was fixed in #85344, this test checks that "auxiliary" is being
// successfully remapped to "/the/aux" in the rmeta files.
// See https://github.com/rust-lang/rust/pull/85344

// FIXME(Oneirical): check if works without ignore-windows

use run_make_support::{invalid_utf8_contains, invalid_utf8_not_contains, is_darwin, rustc};

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
    invalid_utf8_contains("liblib.rmeta", "/the/aux/lib.rs");
    invalid_utf8_not_contains("liblib.rmeta", "auxiliary");

    out_object.arg("-Zremap-path-scope=object");
    out_macro.arg("-Zremap-path-scope=macro");
    out_diagobj.arg("-Zremap-path-scope=diagnostics,object");
    if is_darwin() {
        out_object.arg("-Csplit-debuginfo=off");
        out_macro.arg("-Csplit-debuginfo=off");
        out_diagobj.arg("-Csplit-debuginfo=off");
    }

    out_object.run();
    invalid_utf8_contains("liblib.rmeta", "/the/aux/lib.rs");
    invalid_utf8_not_contains("liblib.rmeta", "auxiliary");
    out_macro.run();
    invalid_utf8_contains("liblib.rmeta", "/the/aux/lib.rs");
    invalid_utf8_not_contains("liblib.rmeta", "auxiliary");
    out_diagobj.run();
    invalid_utf8_contains("liblib.rmeta", "/the/aux/lib.rs");
    invalid_utf8_not_contains("liblib.rmeta", "auxiliary");
}
