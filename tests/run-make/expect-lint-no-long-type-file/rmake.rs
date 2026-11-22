// Regression test for https://github.com/rust-lang/rust/issues/166293.
// An expected lint is never shown, so it must not write a long type file to disk.
// A lint that is shown still must write one, which guards against a vacuous pass.

use run_make_support::{rfs, rustc};

fn long_type_files() -> usize {
    rfs::read_dir(".")
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_name().to_string_lossy().contains("long-type"))
        .count()
}

fn main() {
    rustc()
        .input("lint.rs")
        .edition("2024")
        .arg("-Zwrite-long-types-to-disk=yes")
        .arg("--diagnostic-width=60")
        .run();
    assert_eq!(long_type_files(), 0, "expected lint created a long type file");

    rustc()
        .input("allowed.rs")
        .edition("2024")
        .arg("-Zwrite-long-types-to-disk=yes")
        .arg("--diagnostic-width=60")
        .run();
    assert_eq!(long_type_files(), 1, "shown lint should create a long type file");
}
