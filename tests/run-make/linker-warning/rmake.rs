//@ ignore-cross-compile (need to run fake linker)

use run_make_support::{Rustc, diff, regex, rustc};

fn run_rustc() -> Rustc {
    let mut rustc = rustc();
    rustc
        .arg("main.rs")
        // NOTE: `link-self-contained` can vary depending on bootstrap.toml.
        // Make sure we use a consistent value.
        .arg("-Clink-self-contained=-linker")
        .arg("-Zunstable-options")
        .arg("-Wlinker-messages")
        .args(["--extern", "foo", "--extern", "bar"])
        .output("main")
        .linker("./fake-linker");
    if run_make_support::target() == "x86_64-unknown-linux-gnu" {
        // The value of `rust.lld` is different between CI and locally. Override it explicitly.
        rustc.arg("-Clinker-flavor=gnu-cc");
    }
    rustc
}

fn main() {
    // first, compile our linker and our dependencies
    rustc().arg("fake-linker.rs").output("fake-linker").run();
    rustc().arg("foo.rs").crate_type("rlib").run();
    rustc().arg("bar.rs").crate_type("rlib").run();

    // Run rustc with our fake linker, and make sure it shows warnings
    let warnings = run_rustc().link_arg("run_make_warn").run();
    warnings.assert_stderr_contains("warning: linker stderr: bar");

    // Make sure it shows stdout
    run_rustc()
        .link_arg("run_make_info")
        .run()
        .assert_stderr_contains("warning: linker stdout: foo");

    // Make sure we short-circuit this new path if the linker exits with an error
    // (so the diagnostic is less verbose)
    run_rustc().link_arg("run_make_error").run_fail().assert_stderr_contains("note: error: baz");

    // Make sure we don't show the linker args unless `--verbose` is passed
    let out = run_rustc().link_arg("run_make_error").verbose().run_fail();
    out.assert_stderr_contains_regex("fake-linker.*run_make_error")
        .assert_stderr_not_contains("object files omitted")
        .assert_stderr_contains(r".rcgu.o")
        .assert_stderr_contains_regex(r"lib(/|\\\\)libstd");

    let out = run_rustc().link_arg("run_make_error").run_fail();
    out.assert_stderr_contains("fake-linker")
        .assert_stderr_contains("object files omitted")
        .assert_stderr_contains("/{libfoo,libbar}.rlib\"")
        .assert_stderr_contains("-*}.rlib\"")
        .assert_stderr_not_contains(r".rcgu.o")
        .assert_stderr_not_contains_regex(r"lib(/|\\\\)libstd");

    // FIXME: we should have a version of this for mac and windows
    if run_make_support::target() == "x86_64-unknown-linux-gnu" {
        diff()
            .expected_file("short-error.txt")
            .actual_text("(linker error)", out.stderr())
            .normalize(
                regex::escape(
                    run_make_support::build_root().canonicalize().unwrap().to_str().unwrap(),
                ),
                "/build-root",
            )
            .normalize("libpanic_abort", "libpanic_unwind")
            .normalize(r#""[^"]*\/symbols.o""#, "\"/symbols.o\"")
            .normalize(r#""[^"]*\/raw-dylibs""#, "\"/raw-dylibs\"")
            .run();
    }

    // Make sure a single dependency doesn't use brace expansion.
    let out1 = run_rustc().cfg("only_foo").link_arg("run_make_error").run_fail();
    out1.assert_stderr_contains("fake-linker").assert_stderr_contains("/libfoo.rlib\"");

    // Make sure we show linker warnings even across `-Z no-link`
    rustc()
        .arg("-Zno-link")
        .input("-")
        .stdin_buf("#![deny(linker_messages)] \n fn main() {}")
        .run()
        .assert_stderr_equals("");
    rustc()
        .arg("-Zlink-only")
        .arg("rust_out.rlink")
        .linker("./fake-linker")
        .link_arg("run_make_warn")
        .run_fail()
        // NOTE: the error message here is quite bad (we don't have a source
        // span, but still try to print the lint source). But `-Z link-only` is
        // unstable and this still shows the linker warning itself so this is
        // probably good enough.
        .assert_stderr_contains("linker stderr: bar");

    // Same thing, but with json output.
    rustc()
        .error_format("json")
        .arg("-Zlink-only")
        .arg("rust_out.rlink")
        .linker("./fake-linker")
        .link_arg("run_make_warn")
        .run_fail()
        .assert_stderr_contains(r#""$message_type":"diagnostic""#);
}
