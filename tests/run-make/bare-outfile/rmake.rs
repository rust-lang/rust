// This test checks that manually setting the output file as a bare file with no file extension
// still results in successful compilation.

//@ ignore-cross-compile

use run_make_support::{run, rustc, tmp_dir};
use std::env;
use std::fs;

fn main() {
    fs::copy("foo.rs", tmp_dir().join("foo.rs")).unwrap();
    env::set_current_dir(tmp_dir());
    rustc().output("foo").input("foo.rs").run();
    run("foo");
}
