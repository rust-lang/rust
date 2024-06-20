// Checks that if the output folder doesn't exist, rustdoc will create it.

use run_make_support::rustdoc;
use std::path::Path;

fn main() {
    let out_dir = Path::new("foo/bar/doc");
    rustdoc().input("foo.rs").output(&out_dir).run();
    assert!(out_dir.exists());
}
