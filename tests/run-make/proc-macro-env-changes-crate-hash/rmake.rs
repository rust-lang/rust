// Verifies that when a proc-macro's *output* changes without its source
// changing — here, because the proc-macro reads an environment variable at
// expansion time and we vary that variable between consumer builds — the
// consumer crate's crate_hash / SVH changes.
//
// Companion to `proc-macro-dep-source-changes-crate-hash`: that test covers
// the case where the proc-macro source (and therefore its compiled metadata)
// changes; this test covers the case where only the tokens produced during
// expansion change. Both must invalidate the consumer's crate_hash.
//
// Note: the env var is read at consumer-compile time (when the proc-macro
// runs), so we set it on the `rustc` invocation that builds `foo.rs`, not on
// the one that builds the proc-macro itself.
// See https://github.com/rust-lang/rust/issues/94878 and PR #154724.

//@ needs-crate-type: proc-macro

use run_make_support::{diff, rfs, rustc};

const ENV_VAR: &str = "PROC_MACRO_DEP_TOKEN";

fn build_in(dir: &str, value: &str) -> String {
    // The proc-macro is built once per build, but its source is identical;
    // what differs is the value of ENV_VAR seen during expansion in the
    // consumer build.
    rustc()
        .input("changing_macro.rs")
        .crate_name("changing_macro")
        .crate_type("proc-macro")
        .out_dir(dir)
        .run();
    rustc().input("foo.rs").library_search_path(dir).out_dir(dir).env(ENV_VAR, value).run();
    rustc().arg("-Zls=root").input(format!("{dir}/libfoo.rlib")).run().stdout_utf8()
}

fn main() {
    rfs::create_dir("a");
    rfs::create_dir("b");
    rfs::create_dir("a_again");

    // Build the consumer twice with the same proc-macro source but different
    // values of ENV_VAR. foo.rs is byte-identical; only the tokens spliced in
    // by the derive differ.
    let a = build_in("a", "first");
    let b = build_in("b", "second");
    // The SVH (printed by `-Zls=root`) must differ between the two builds.
    diff().expected_text("a", &a).actual_text("b", b).run_fail();

    // Sanity: rebuilding with the original env value reproduces the original
    // dump, so the difference above is genuinely caused by the env change and
    // not by non-determinism in metadata encoding.
    let a_again = build_in("a_again", "first");
    diff().expected_text("a", &a).actual_text("a_again", a_again).run();
}
