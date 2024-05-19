// This test checks that C linking with Rust does not encounter any errors, with a static library.
// See https://github.com/rust-lang/rust/issues/10434

//@ ignore-cross-compile

use run_make_support::{cc, extra_c_flags, run, rustc, static_lib};
use std::fs;

fn main() {
    rustc().input("foo.rs").run();
    cc().input("bar.c").input(static_lib("foo")).out_exe("bar").args(&extra_c_flags()).run();
    run("bar");
    fs::remove_file(static_lib("foo"));
    run("bar");
}
