//@ needs-target-std
//
// A dependency's `extra_filename` is only a hint for locating it on disk, not
// part of its identity: the crate loader treats two libraries with equal SVHs
// as the same crate. Verifies that changing the `extra_filename` of a
// dependency does not change the crate hash of its dependents.
// See https://github.com/rust-lang/rust/issues/94878 and PR #154724.

use run_make_support::{diff, rfs, rustc};

fn build_a(dir: &str, extra_filename: &str) -> String {
    // Build `a` into `dir` under the given `-C extra-filename`, and return the
    // path of the rlib it produced.
    let mut cmd = rustc();
    cmd.input("a.rs").crate_name("a").crate_type("rlib").out_dir(dir);
    if !extra_filename.is_empty() {
        cmd.arg(format!("-Cextra-filename={extra_filename}"));
    }
    cmd.run();
    format!("{dir}/liba{extra_filename}.rlib")
}

fn build_b(dir: &str, a_rlib: &str) -> String {
    // Build `b` against the given build of `a`, then dump `b`'s SVH. Every
    // build of `b` is invoked identically; only which `a` it links differs.
    rustc()
        .input("b.rs")
        .crate_name("b")
        .crate_type("rlib")
        .extern_("a", a_rlib)
        .out_dir(dir)
        .run();
    svh(&format!("{dir}/libb.rlib"))
}

fn svh(rlib: &str) -> String {
    // Only the `hash` line of the `-Zls=root` dump. The rest of the dump lists
    // the dependencies, and their `extra_filename`s are exactly what varies
    // here, so comparing it whole would differ for uninteresting reasons.
    let dump = rustc().arg("-Zls=root").input(rlib).run().stdout_utf8();
    dump.lines()
        .find(|line| line.starts_with("hash "))
        .expect("`-Zls=root` printed no `hash` line")
        .to_owned()
}

fn main() {
    rfs::create_dir("a-one");
    rfs::create_dir("a-two");
    rfs::create_dir("a-here");
    rfs::create_dir("a-there");
    rfs::create_dir("b-one");
    rfs::create_dir("b-two");
    rfs::create_dir("b-here");
    rfs::create_dir("b-there");
    rfs::create_dir("b-one-again");

    // Two builds of `a` from byte-identical sources, differing only in
    // `-C extra-filename`.
    let a_one = build_a("a-one", "-one");
    let a_two = build_a("a-two", "-two");

    // A crate's own `extra_filename` is already excluded from its own SVH, so
    // the two builds of `a` must hash identically. Without this the rest of
    // the test would prove nothing, because `b` would be entitled to change
    // on account of `a` having changed.
    let a_one_svh = svh(&a_one);
    diff().expected_text("a-one", &a_one_svh).actual_text("a-two", svh(&a_two)).run();

    // The property under test: `a`'s `extra_filename` must not reach `b`'s SVH.
    let b_one = build_b("b-one", &a_one);
    let b_two = build_b("b-two", &a_two);
    diff().expected_text("b-one", &b_one).actual_text("b-two", b_two).run();

    // Those two builds also read `a` from different directories, so pin that a
    // dependency's *path* does not move `b`'s SVH either. This is what makes
    // the comparison above attributable to `extra_filename`.
    let a_here = build_a("a-here", "");
    let a_there = build_a("a-there", "");
    let b_here = build_b("b-here", &a_here);
    let b_there = build_b("b-there", &a_there);
    diff().expected_text("b-here", &b_here).actual_text("b-there", b_there).run();

    // Sanity: rebuilding `b` against the original `a` reproduces its SVH, so
    // the comparisons above are not passing by accident of metadata encoding
    // being non-deterministic.
    let b_one_again = build_b("b-one-again", &a_one);
    diff().expected_text("b-one", &b_one).actual_text("b-one-again", b_one_again).run();
}
