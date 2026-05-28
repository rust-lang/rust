// This test intentionally feeds invalid inputs to codegen and checks if the error message outputs
// contain specific helpful indications.

//@ ignore-cross-compile

use run_make_support::regex::Regex;
use run_make_support::rustc;

fn main() {
    // Option taking a number.
    rustc()
        .input("dummy.rs")
        .arg("-Ccodegen-units")
        .run_fail()
        .assert_stderr_contains("codegen option `codegen-units` requires a number");
    rustc().input("dummy.rs").arg("-Ccodegen-units=").run_fail().assert_stderr_contains(
        "incorrect value `` for codegen option `codegen-units` - a number was expected",
    );
    rustc().input("dummy.rs").arg("-Ccodegen-units=foo").run_fail().assert_stderr_contains(
        "incorrect value `foo` for codegen option `codegen-units` - a number was expected",
    );
    rustc().input("dummy.rs").arg("-Ccodegen-units=1").run();

    // Option taking a string.
    rustc()
        .input("dummy.rs")
        .arg("-Cextra-filename")
        .run_fail()
        .assert_stderr_contains("codegen option `extra-filename` requires a string");
    rustc().input("dummy.rs").arg("-Cextra-filename=").run();
    rustc().input("dummy.rs").arg("-Cextra-filename=foo").run();

    // Option taking no argument.
    rustc().input("dummy.rs").arg("-Clto=").run_fail().assert_stderr_contains(
        "codegen option `lto` - either a boolean (`yes`, `no`, `on`, `off`, etc), `thin`, \
             `fat`, or omitted",
    );
    rustc().input("dummy.rs").arg("-Clto=1").run_fail().assert_stderr_contains(
        "codegen option `lto` - either a boolean (`yes`, `no`, `on`, `off`, etc), `thin`, \
             `fat`, or omitted",
    );
    rustc().input("dummy.rs").arg("-Clto=foo").run_fail().assert_stderr_contains(
        "codegen option `lto` - either a boolean (`yes`, `no`, `on`, `off`, etc), `thin`, \
             `fat`, or omitted",
    );
    rustc().input("dummy.rs").arg("-Clto").run();

    let regex = Regex::new("--gc-sections|-z[^ ]* [^ ]*<ignore>|-dead_strip|/OPT:REF").unwrap();
    // Should not link dead code...
    let stdout = rustc().input("dummy.rs").print("link-args").run().stdout_utf8();
    assert!(regex.is_match(&stdout));
    // ... unless you specifically ask to keep it
    let stdout =
        rustc().input("dummy.rs").print("link-args").arg("-Clink-dead-code").run().stdout_utf8();
    assert!(!regex.is_match(&stdout));
}
