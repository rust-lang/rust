use run_make_support::fs_wrapper::create_file;
use run_make_support::{ar, rustc};

fn main() {
    create_file("lib.rmeta");
    ar(&["lib.rmeta"], "libfoo-ffffffff-1.0.rlib");
    rustc().input("foo.rs").run_fail().assert_stderr_contains("found invalid metadata");
}
