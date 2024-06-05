// This test creates some fake dynamic libraries with nothing inside,
// and checks if rustc avoids them and successfully compiles as a result.

//@ ignore-cross-compile

use run_make_support::{dynamic_lib, rustc};
use std::fs::File;

fn main() {
    rustc().input("foo.rs").arg("-Cprefer-dynamic").run();
    File::create(dynamic_lib("foo-something-special")).unwrap();
    File::create(dynamic_lib("foo-something-special2")).unwrap();
    rustc().input("bar.rs").run();
}
