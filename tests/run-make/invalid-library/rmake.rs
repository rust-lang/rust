use run_make_support::fs_wrapper::create_file;
use run_make_support::{ar_command, rustc};

fn main() {
    create_file("lib.rmeta");
    ar_command().arg("crus").arg("libfoo-ffffffff-1.0.rlib").arg("lib.rmeta").run();
    rustc().input("foo.rs").run_fail().assert_stderr_contains("found invalid metadata");
}
