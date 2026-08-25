// Checks that two dylibs compiled with code coverage enabled can be linked
// together without getting an error about duplicate profiler_builtins.

//@ needs-profiler-runtime
//@ needs-dynamic-linking

use run_make_support::{dynamic_lib_name, rustc};

fn main() {
    rustc()
        .crate_name("dylib_a")
        .crate_type("dylib")
        .arg("-Cinstrument-coverage")
        .arg("-Cprefer-dynamic")
        .input("dylib_a.rs")
        .run();
    rustc()
        .crate_name("dylib_b")
        .crate_type("dylib")
        .arg("-Cinstrument-coverage")
        .arg("-Cprefer-dynamic")
        .input("dylib_b.rs")
        .run();
    rustc()
        .crate_type("bin")
        .extern_("dylib_a", dynamic_lib_name("dylib_a"))
        .extern_("dylib_b", dynamic_lib_name("dylib_b"))
        .input("main.rs")
        .run();
}
