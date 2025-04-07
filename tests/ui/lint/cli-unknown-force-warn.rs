// Checks that rustc correctly errors when passed an invalid lint with
// `--force-warn`. This is a regression test for issue #86958.

//@ check-pass
//@ compile-flags: --force-warn foo-qux

//@ error-pattern: requested on the command line with `--force-warn foo_qux`
//@ error-pattern: `#[warn(unknown_lints)]` on by default

fn main() {}

//~? WARN unknown lint: `foo_qux`
//~? WARN unknown lint: `foo_qux`
//~? WARN unknown lint: `foo_qux`
