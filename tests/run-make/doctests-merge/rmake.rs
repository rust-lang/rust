extern crate run_make_support;

use run_make_support::{diff, rustc, rustdoc, tmp_dir};
use std::path::Path;

fn test_and_compare(
    input_file: &str,
    stdout_file: &str,
    edition: &str,
    should_succeed: bool,
    dep: &Path,
) {
    let mut cmd = rustdoc();

    cmd.input(input_file)
        .arg("--test")
        .arg("-Zunstable-options")
        .arg("--edition")
        .arg(edition)
        .arg("--test-args=--test-threads=1")
        .arg("--extern")
        .arg(format!("foo={}", dep.display()))
        .env("RUST_BACKTRACE", "short");
    let output = if should_succeed { cmd.run() } else { cmd.run_fail() };

    diff()
        .expected_file(stdout_file)
        .actual_text("output", output.stdout)
        .normalize(r#"finished in \d+\.\d+s"#, "finished in $$TIME")
        .run();
}

fn main() {
    let out_file = tmp_dir().join("libfoo.rlib");

    rustc().input("doctest.rs").crate_type("rlib").arg("-o").arg(&out_file).run();

    // First we ensure that running with the 2024 edition will fail at runtime.
    test_and_compare("doctest.rs", "doctest-failure.stdout", "2024", false, &out_file);

    // Then we ensure that running with an edition < 2024 will not fail at runtime.
    test_and_compare("doctest.rs", "doctest-success.stdout", "2021", true, &out_file);

    // Now we check with the standalone attribute which should succeed in all cases.
    test_and_compare("doctest-standalone.rs", "doctest-standalone.stdout", "2024", true, &out_file);
    test_and_compare("doctest-standalone.rs", "doctest-standalone.stdout", "2021", true, &out_file);
}
