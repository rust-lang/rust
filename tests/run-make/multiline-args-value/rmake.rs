use run_make_support::{cwd, diff, rustc};

fn test_and_compare(test_name: &str, flag: &str, val: &str) {
    let mut cmd = rustc();

    let output = cmd.input("").arg("--crate-type=lib").arg(flag).arg(val).run_fail();

    assert_eq!(output.stdout_utf8(), "");
    diff()
        .expected_file(format!("{test_name}.stderr"))
        .actual_text("stderr", output.stderr_utf8())
        .run();
}

fn main() {
    // Verify that frontmatter isn't allowed in `--cfg` arguments.
    // https://github.com/rust-lang/rust/issues/146130
    test_and_compare(
        "cfg-frontmatter",
        "--cfg",
        r#"---
---
key"#,
    );

    // Verify that frontmatter isn't allowed in `--check-cfg` arguments.
    // https://github.com/rust-lang/rust/issues/146130
    test_and_compare(
        "check-cfg-frontmatter",
        "--check-cfg",
        r#"---
---
cfg(key)"#,
    );

    // Verify that shebang isn't allowed in `--cfg` arguments.
    test_and_compare(
        "cfg-shebang",
        "--cfg",
        r#"#!/usr/bin/shebang
key"#,
    );

    // Verify that shebang isn't allowed in `--check-cfg` arguments.
    test_and_compare(
        "check-cfg-shebang",
        "--check-cfg",
        r#"#!/usr/bin/shebang
cfg(key)"#,
    );
}
