//@ needs-target-std
//
// In this test, the function `bar` has #[inline(never)] and the function `foo`
// does not. This test outputs LLVM optimization remarks twice - first for all
// functions (including `bar`, and the `inline` mention), and then for only `foo`
// (should not have the `inline` mention).
// See https://github.com/rust-lang/rust/pull/113040

use run_make_support::{
    has_extension, has_prefix, invalid_utf8_contains, invalid_utf8_not_contains, not_contains,
    rustc, shallow_find_files,
};

fn main() {
    rustc()
        .opt()
        .input("foo.rs")
        .crate_type("lib")
        .arg("-Cremark=all")
        .arg("-Zremark-dir=profiles_all")
        .run();
    let all_remark_files = shallow_find_files("profiles_all", |path| {
        has_prefix(path, "foo") && has_extension(path, "yaml") && not_contains(path, "codegen")
    });
    for file in all_remark_files {
        invalid_utf8_contains(file, "inline")
    }
    rustc()
        .opt()
        .input("foo.rs")
        .crate_type("lib")
        .arg("-Cremark=foo")
        .arg("-Zremark-dir=profiles_foo")
        .run();
    let foo_remark_files = shallow_find_files("profiles_foo", |path| {
        has_prefix(path, "foo") && has_extension(path, "yaml")
    });
    for file in foo_remark_files {
        invalid_utf8_not_contains(file, "inline")
    }
}
