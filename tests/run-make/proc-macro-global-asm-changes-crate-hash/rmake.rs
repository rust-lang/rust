// Verifies that when a proc-macro emits a `global_asm!` block whose template
// changes between builds (here driven by an env var read at expansion time),
// the consumer crate's crate_hash / SVH changes.
//
// This exercises an item kind (`DefKind::GlobalAsm`) whose body lives only
// in HIR and is *not* recorded in any way by the metadata encoder:
// `should_encode_span`, `should_encode_attrs`, `should_encode_visibility`,
// `should_encode_generics`, `should_encode_type` and `should_encode_mir` are
// all false for `GlobalAsm`, and `def_kind.has_codegen_attrs()` is false too
// (see `compiler/rustc_metadata/src/rmeta/encoder.rs`). All that ends up in
// the rmeta byte stream for a `global_asm!` invocation is the fixed-size
// `DefKind::GlobalAsm` enum discriminant in the def_kind table, which is
// identical regardless of the asm template's contents. The asm template
// itself is only read out of HIR later by `MonoItem::GlobalAsm` codegen in
// `rustc_monomorphize::collector`.
//
// Companion to `proc-macro-dep-source-changes-crate-hash` and
// `proc-macro-env-changes-crate-hash`.
// See https://github.com/rust-lang/rust/issues/94878 and PR #154724.

//@ needs-crate-type: proc-macro

use run_make_support::{diff, rfs, rustc};

const ENV_VAR: &str = "PROC_MACRO_ASM_TOKEN";

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
    // values of ENV_VAR. foo.rs is byte-identical; only the asm template
    // spliced in by `emit_global_asm!` differs.
    let a = build_in("a", "first");
    let b = build_in("b", "second");
    // The SVH (printed by `-Zls=root`) must differ between the two builds.
    // Under PR #154724 without an HIR-hash contribution, the rmeta encoder
    // writes no bytes that depend on the asm template, so the two SVHs are
    // identical and this `run_fail` fails (the dumps match).
    diff().expected_text("a", &a).actual_text("b", b).run_fail();

    // Sanity: rebuilding with the original env value reproduces the original
    // dump, so the difference above is genuinely caused by the env change
    // and not by non-determinism in metadata encoding.
    let a_again = build_in("a_again", "first");
    diff().expected_text("a", &a).actual_text("a_again", a_again).run();
}
