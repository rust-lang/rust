// Checks that rustc correctly errors when passed an invalid lint with
// `--force-warn`. This is a regression test for issue #86958.

//@ check-pass
//@ compile-flags: --force-warn foo-qux
//@ dont-require-annotations: NOTE

fn main() {}

//~? WARN unknown lint: `foo_qux`
//~? WARN unknown lint: `foo_qux`
//~? WARN unknown lint: `foo_qux`
//~? NOTE requested on the command line with `--force-warn foo_qux`
//~? NOTE `#[warn(unknown_lints)]` on by default
