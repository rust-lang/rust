// This test verifies that rustdoc `-o -` prints JSON on stdout and doesn't generate
// a JSON file.

use std::path::PathBuf;

use run_make_support::path_helpers::{cwd, has_extension, read_dir_entries_recursive};
use run_make_support::rustdoc;

fn main() {
    // First we check that we generate the JSON in the stdout.
    rustdoc()
        .input("foo.rs")
        .out_dir("-")
        .arg("-Zunstable-options")
        .output_format("json")
        .run()
        .assert_stdout_contains("{\"");

    // Then we check it didn't generate any JSON file.
    read_dir_entries_recursive(cwd(), |path| {
        if path.is_file() && has_extension(path, "json") {
            panic!("Found a JSON file {path:?}");
        }
    });
}
