//@ ignore-cross-compile (needs to run doctests)

use std::path::Path;

use run_make_support::{cwd, diff, rustc, rustdoc};

fn test_and_compare(input_file: &str, stdout_file: &str, edition: &str, dep: &Path) {
    let mut cmd = rustdoc();

    let output = cmd
        .input(input_file)
        .arg("--test")
        .edition(edition)
        .arg("--test-args=--test-threads=1")
        .extern_("foo", dep.display().to_string())
        .env("RUST_BACKTRACE", "short")
        .run();

    diff()
        .expected_file(stdout_file)
        .actual_text("output", output.stdout_utf8())
        .normalize(r#"finished in \d+\.\d+s"#, "finished in $$TIME")
        .run();
}

fn main() {
    let out_file = cwd().join("libfoo.rlib");

    rustc().input("doctest.rs").crate_type("rlib").output(&out_file).run();

    // First we ensure that running with the 2024 edition will not fail at runtime.
    test_and_compare("doctest.rs", "doctest-2024.stdout", "2024", &out_file);

    // Then we ensure that running with an edition < 2024 will not fail at runtime.
    test_and_compare("doctest.rs", "doctest-2021.stdout", "2021", &out_file);

    // Now we check with the standalone attribute which should succeed in all cases.
    test_and_compare("doctest-standalone.rs", "doctest-standalone.stdout", "2024", &out_file);
    test_and_compare("doctest-standalone.rs", "doctest-standalone.stdout", "2021", &out_file);
}
