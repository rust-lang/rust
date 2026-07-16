// This test ensures that `-o` option works as expected with `--show-coverage`.
// Regression test for <https://github.com/rust-lang/rust/issues/158929>.

use run_make_support::rfs::{read_to_string, remove_file};
use run_make_support::{path, rustdoc};

fn run_rustdoc(extra_args: &[&str]) -> String {
    rustdoc()
        .input("foo.rs")
        .arg("-Zunstable-options")
        .arg("--show-coverage")
        .args(extra_args)
        .run()
        .stdout_utf8()
}

fn check_print_stdout(extra_args: &[&str], stdout_check: &str) {
    let out = run_rustdoc(extra_args);

    // By default, it shouldn't have created a `doc` folder.
    assert!(!path("doc").exists(), "`doc` folder created with {extra_args:?}");
    // It should have display its output on stdout.
    assert!(out.starts_with(stdout_check), "{out:?} doesn't start with {stdout_check:?}");
}

fn check_generate_file(ext: &str, extra_args: &[&str], file_check: &str) {
    let mut args = extra_args.to_vec();
    args.push("-o");
    args.push("doc");
    let out = run_rustdoc(&args);

    // By default, it shouldn't have created a `doc` folder.
    assert!(path("doc").exists(), "`doc` folder not created with {args:?}");
    let file = format!("doc/foo.{ext}");
    assert!(path(&file).exists());

    let expected = format!("Generated output into {file:?}\n");
    assert_eq!(out, expected, "Expected {expected:?}, got {out:?}");

    let content = read_to_string(&file);
    assert!(content.starts_with(file_check), "{content:?} doesn't start with {file_check:?}");
    remove_file(file);
}

fn main() {
    check_print_stdout(&[], "+-");
    check_print_stdout(&["-o", "-"], "+-");
    check_print_stdout(&["--output-format=json"], "{");
    check_print_stdout(&["--output-format=json", "-o", "-"], "{");

    // Now we check that it works with "-o something".
    check_generate_file("txt", &[], "+-");
    check_generate_file("json", &["--output-format=json"], "{");
}
