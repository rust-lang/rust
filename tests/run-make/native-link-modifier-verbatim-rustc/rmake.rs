//@ needs-target-std
//
// `verbatim` is a native link modifier that forces rustc to only accept libraries with
// a specified name. This test checks that this modifier works as intended.
// This test is the same as native-link-modifier-linker, but with rlibs.
// See https://github.com/rust-lang/rust/issues/99425

use run_make_support::rustc;

fn main() {
    // Verbatim allows for the specification of a precise name
    // - in this case, the unconventional ".ext" extension.
    rustc()
        .input("upstream_native_dep.rs")
        .crate_type("staticlib")
        .output("upstream_some_strange_name.ext")
        .run();
    rustc()
        .input("rust_dep.rs")
        .crate_type("rlib")
        .arg("-lstatic:+verbatim=upstream_some_strange_name.ext")
        .run();

    // This section voluntarily avoids using static_lib_name helpers to be verbatim.
    // With verbatim, even these common library names are refused
    // - it wants upstream_native_dep without
    // any file extensions.
    rustc()
        .input("upstream_native_dep.rs")
        .crate_type("staticlib")
        .output("libupstream_native_dep.a")
        .run();
    rustc()
        .input("upstream_native_dep.rs")
        .crate_type("staticlib")
        .output("upstream_native_dep.a")
        .run();
    rustc()
        .input("upstream_native_dep.rs")
        .crate_type("staticlib")
        .output("upstream_native_dep.lib")
        .run();
    rustc()
        .input("rust_dep.rs")
        .crate_type("rlib")
        .arg("-lstatic:+verbatim=upstream_native_dep")
        .run_fail()
        .assert_stderr_contains("upstream_native_dep");
}
