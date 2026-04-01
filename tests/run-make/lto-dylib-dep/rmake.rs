// Compiling with link-time-optimizations (LTO) would previously run into an internal
// compiler error (ICE) if a dylib was passed as a required library. This was due to a
// misplaced assert! call in the compiler, which is now removed. This test checks that
// this bug does not make a resurgence and that dylib+lto compilation succeeds.
// See https://github.com/rust-lang/rust/issues/59137

//@ ignore-cross-compile

use run_make_support::{run, rustc};

fn main() {
    rustc().input("a_dylib.rs").crate_type("dylib").arg("-Cprefer-dynamic").run();
    rustc().input("main.rs").arg("-Clto").run();
    run("main");
}
