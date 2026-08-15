// This test verifies that rustdoc `-o -` prints JSON on stdout and doesn't generate
// a JSON file.

//@ needs-target-std

use run_make_support::path_helpers::{cwd, has_extension, read_dir_entries_recursive};
use run_make_support::{rustdoc, serde_json};

fn main() {
    let json_string = rustdoc()
        .input("foo.rs")
        .out_dir("-")
        .arg("-Zunstable-options")
        .output_format("json")
        .run()
        .stdout_utf8();

    // First we check that we generate the JSON in the stdout.
    let json_value: serde_json::Value =
        serde_json::from_str(&json_string).expect("stdout should be valid json");

    // We don't care to test the specifics of the JSON, as that's done
    // elsewhere, just check that it has a format_version (as all JSON output
    // should).
    let format_version = json_value["format_version"]
        .as_i64()
        .expect("json output should contain format_version field");
    assert!(format_version > 30);

    // Then we check it didn't generate any JSON file.
    read_dir_entries_recursive(cwd(), |path| {
        if path.is_file() && has_extension(path, "json") {
            panic!("Found a JSON file {path:?}");
        }
    });
}
