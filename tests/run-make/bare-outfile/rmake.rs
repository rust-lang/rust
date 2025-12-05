// This test checks that manually setting the output file as a bare file with no file extension
// still results in successful compilation.

//@ ignore-cross-compile

use run_make_support::{run, rustc};

fn main() {
    rustc().output("foo").input("foo.rs").run();
    run("foo");
}
