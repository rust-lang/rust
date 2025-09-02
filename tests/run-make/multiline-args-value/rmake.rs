use run_make_support::{cwd, diff, rustc};

fn test_and_compare(flag: &str, val: &str) {
    let mut cmd = rustc();

    let output =
        cmd.input("").arg("--crate-type=lib").arg(&format!("--{flag}")).arg(val).run_fail();

    assert_eq!(output.stdout_utf8(), "");
    diff()
        .expected_file(format!("{flag}.stderr"))
        .actual_text("output", output.stderr_utf8())
        .run();
}

fn main() {
    // Verify that frontmatter isn't allowed in `--cfg` arguments.
    // https://github.com/rust-lang/rust/issues/146130
    test_and_compare(
        "cfg",
        r#"---
---
key"#,
    );

    // Verify that frontmatter isn't allowed in `--check-cfg` arguments.
    // https://github.com/rust-lang/rust/issues/146130
    test_and_compare(
        "check-cfg",
        r#"---
---
cfg(key)"#,
    );
}
