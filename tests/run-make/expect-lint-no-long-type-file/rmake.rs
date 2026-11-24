// Regression test for https://github.com/rust-lang/rust/issues/166293.
// Expected and allowed lints are never shown, so they must not write a long type file to disk.
// A lint that is shown still must write one, which guards against a vacuous pass.

use run_make_support::{rfs, rustc};

/// Compiles `input` and returns the long type files it created, removing them afterwards so
/// that every case is counted independently.
fn compile_and_count_long_type_files(input: &str) -> usize {
    rustc()
        .input(input)
        .edition("2024")
        .arg("-Zwrite-long-types-to-disk=yes")
        .arg("--diagnostic-width=60")
        .run();

    let files: Vec<_> = rfs::read_dir(".")
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_name().to_string_lossy().contains("long-type"))
        .collect();
    for file in &files {
        rfs::remove_file(file.path());
    }
    files.len()
}

fn main() {
    assert_eq!(
        compile_and_count_long_type_files("expected.rs"),
        0,
        "expected lint created a long type file"
    );
    assert_eq!(
        compile_and_count_long_type_files("allowed.rs"),
        0,
        "allowed lint created a long type file"
    );
    assert_eq!(
        compile_and_count_long_type_files("shown.rs"),
        1,
        "shown lint should create a long type file"
    );
}
