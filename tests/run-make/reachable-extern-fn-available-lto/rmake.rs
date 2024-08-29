// Test to make sure that reachable extern fns are always available in final
// productcs, including when link time optimizations (LTO) are used.

// In this test, the `foo` crate has a reahable symbol,
// and is a dependency of the `bar` crate. When the `bar` crate
// is compiled with LTO, it shouldn't strip the symbol from `foo`, and that's the
// only way that `foo.c` will successfully compile.
// See https://github.com/rust-lang/rust/issues/14500

//@ ignore-cross-compile

use run_make_support::{cc, extra_c_flags, run, rustc, static_lib_name};

fn main() {
    let libbar_path = static_lib_name("bar");
    rustc().input("foo.rs").crate_type("rlib").run();
    rustc()
        .input("bar.rs")
        .crate_type("staticlib")
        .arg("-Clto")
        .library_search_path(".")
        .output(&libbar_path)
        .run();
    cc().input("foo.c").input(libbar_path).args(&extra_c_flags()).out_exe("foo").run();
    run("foo");
}
