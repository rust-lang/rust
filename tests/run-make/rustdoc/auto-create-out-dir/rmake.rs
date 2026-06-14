// Checks that if the output folder doesn't exist, rustdoc will create it.

//@ needs-target-std

use run_make_support::{path, rustdoc};

fn main() {
    let out_dir = path("foo/bar/doc");
    rustdoc().input("foo.rs").out_dir(&out_dir).run();
    assert!(out_dir.exists());
}
