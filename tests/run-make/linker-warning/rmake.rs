use run_make_support::{Rustc, diff, regex, rustc};

fn run_rustc() -> Rustc {
    let mut rustc = rustc();
    rustc
        .arg("main.rs")
        // NOTE: `link-self-contained` can vary depending on config.toml.
        // Make sure we use a consistent value.
        .arg("-Clink-self-contained=-linker")
        .arg("-Clinker-flavor=gnu-cc")
        .arg("-Zunstable-options")
        .output("main")
        .linker("./fake-linker");
    rustc
}

fn main() {
    // first, compile our linker
    rustc().arg("fake-linker.rs").output("fake-linker").run();

    // Make sure we don't show the linker args unless `--verbose` is passed
    let out = run_rustc().link_arg("run_make_error").verbose().run_fail();
    out.assert_stderr_contains_regex("fake-linker.*run_make_error")
        .assert_stderr_not_contains("object files omitted")
        .assert_stderr_contains("PATH=\"")
        .assert_stderr_contains_regex(r"lib(/|\\\\)libstd");

    let out = run_rustc().link_arg("run_make_error").run_fail();
    out.assert_stderr_contains("fake-linker")
        .assert_stderr_contains("object files omitted")
        .assert_stderr_contains_regex(r"\{")
        .assert_stderr_not_contains("PATH=\"")
        .assert_stderr_not_contains_regex(r"lib(/|\\\\)libstd");

    // FIXME: we should have a version of this for mac and windows
    if run_make_support::target() == "x86_64-unknown-linux-gnu" {
        diff()
            .expected_file("short-error.txt")
            .actual_text("(linker error)", out.stderr())
            .normalize(r#"/rustc[^/]*/"#, "/rustc/")
            .normalize(
                regex::escape(run_make_support::build_root().to_str().unwrap()),
                "/build-root",
            )
            .run();
    }
}
