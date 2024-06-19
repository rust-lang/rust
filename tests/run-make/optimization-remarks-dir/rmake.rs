// In this test, the function `bar` has #[inline(never)] and the function `foo`
// does not. This test outputs LLVM optimization remarks twice - first for all
// functions (including `bar`, and the `inline` mention), and then for only `foo`
// (should not have the `inline` mention).
// See https://github.com/rust-lang/rust/pull/113040

use run_make_support::{invalid_utf8_contains, invalid_utf8_not_contains, rustc};

fn main() {
    rustc()
        .opt()
        .input("foo.rs")
        .crate_type("lib")
        .arg("-Cremark=all")
        .arg("-Zremark-dir=profiles_all")
        .run();
    invalid_utf8_contains("profiles_all/foo.5be5606e1f6aa79b-cgu.0.opt.opt.yaml", "inline");
    rustc()
        .opt()
        .input("foo.rs")
        .crate_type("lib")
        .arg("-Cremark=foo")
        .arg("-Zremark-dir=profiles_foo")
        .run();
    invalid_utf8_not_contains("profiles_foo/foo.5be5606e1f6aa79b-cgu.0.opt.opt.yaml", "inline");
}
