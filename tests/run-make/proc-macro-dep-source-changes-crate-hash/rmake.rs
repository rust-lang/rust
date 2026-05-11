// Verifies that when the *source* of a proc-macro dependency changes (so the
// tokens it emits in the consumer crate change), the consumer crate's
// crate_hash / SVH changes.
// See https://github.com/rust-lang/rust/issues/94878 and PR #154724.

//@ needs-crate-type: proc-macro

use run_make_support::{diff, rfs, rustc};

fn build_in(dir: &str, macro_src: &str) -> String {
    // Build the proc-macro and the consumer into `dir`, then dump the
    // consumer's crate metadata root (which includes the SVH).
    rustc()
        .input(macro_src)
        .crate_name("changing_macro")
        .crate_type("proc-macro")
        .out_dir(dir)
        .run();
    rustc().input("foo.rs").library_search_path(dir).out_dir(dir).run();
    rustc().arg("-Zls=root").input(format!("{dir}/libfoo.rlib")).run().stdout_utf8()
}

fn main() {
    rfs::create_dir("v1");
    rfs::create_dir("v2");
    rfs::create_dir("v1_again");

    // Build the consumer against proc-macro v1, then against v2. foo.rs is
    // byte-identical across builds; only the tokens spliced in by the derive
    // differ.
    let v1 = build_in("v1", "v1.rs");
    let v2 = build_in("v2", "v2.rs");
    // The SVH (printed by `-Zls=root`) must differ between the two builds.
    diff().expected_text("v1", &v1).actual_text("v2", v2).run_fail();

    // Sanity: rebuilding against v1 reproduces the original dump, so the
    // difference above is genuinely caused by the proc-macro source change
    // and not by non-determinism in metadata encoding.
    let v1_again = build_in("v1_again", "v1.rs");
    diff().expected_text("v1", &v1).actual_text("v1_again", v1_again).run();
}
