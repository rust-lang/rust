// A path which contains the same rlib or dylib in two locations
// should not cause an assertion panic in the compiler.
// This test tries to replicate the linked issue and checks
// if the bugged error makes a resurgence.

// See https://github.com/rust-lang/rust/issues/11908

//@ ignore-cross-compile

use run_make_support::{dynamic_lib, rust_lib, rustc, tmp_dir};
use std::fs;

fn main() {
    let tmp_dir_other = tmp_dir().join("other");

    fs::create_dir(&tmp_dir_other);
    rustc().input("foo.rs").crate_type("dylib").arg("-Cprefer-dynamic").run();
    fs::rename(dynamic_lib("foo"), &tmp_dir_other);
    rustc().input("foo.rs").crate_type("dylib").arg("-Cprefer-dynamic").run();
    rustc().input("bar.rs").library_search_path(&tmp_dir_other).run();
    fs::remove_dir_all(tmp_dir());

    fs::create_dir_all(&tmp_dir_other);
    rustc().input("foo.rs").crate_type("rlib").run();
    fs::rename(rust_lib("foo"), &tmp_dir_other);
    rustc().input("foo.rs").crate_type("rlib").run();
    rustc().input("bar.rs").library_search_path(tmp_dir_other).run();
}
