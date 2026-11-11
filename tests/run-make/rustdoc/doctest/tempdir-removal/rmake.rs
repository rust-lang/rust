// This test ensures that no temporary folder is "left behind" when doctests fail for any reason.

//@ ignore-cross-compile

use std::path::Path;

use run_make_support::{path, rfs, rustdoc};

fn run_doctest_and_check_tmpdir(tmp_dir: &Path, doctest: &str, edition: &str) {
    let mut runner = rustdoc();
    runner.input(doctest).arg("--test").edition(edition);
    let output = if cfg!(unix) {
        runner.env("TMPDIR", tmp_dir)
    } else if cfg!(windows) {
        runner.env("TEMP", tmp_dir).env("TMP", tmp_dir)
    } else {
        panic!("unsupported OS")
    }
    .run_fail();

    output.assert_exit_code(101).assert_stdout_contains(
        "test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out",
    );

    rfs::read_dir_entries(tmp_dir, |entry| {
        panic!("Found an item inside the temporary folder: {entry:?}");
    });
}

fn run_doctest_and_check_tmpdir_for_edition(tmp_dir: &Path, edition: &str) {
    run_doctest_and_check_tmpdir(tmp_dir, "compile-error.rs", edition);
    run_doctest_and_check_tmpdir(tmp_dir, "run-error.rs", edition);
}

fn main() {
    let tmp_dir = path("tmp");
    rfs::create_dir(&tmp_dir);

    run_doctest_and_check_tmpdir_for_edition(&tmp_dir, "2018");
    // We use the 2024 edition to check that it's also working for merged doctests.
    run_doctest_and_check_tmpdir_for_edition(&tmp_dir, "2024");
}
