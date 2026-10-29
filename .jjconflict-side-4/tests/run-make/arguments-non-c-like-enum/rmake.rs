//! Check that non-trivial `repr(C)` enum in Rust has valid C layout.
//@ ignore-cross-compile

use run_make_support::{cc, extra_c_flags, extra_cxx_flags, run, rustc, static_lib_name};

pub fn main() {
    rustc().input("nonclike.rs").crate_type("staticlib").run();
    cc().input("test.c")
        .input(static_lib_name("nonclike"))
        .out_exe("test")
        .args(extra_c_flags())
        .args(extra_cxx_flags())
        .inspect(|cmd| eprintln!("{cmd:?}"))
        .run();
    run("test");
}
