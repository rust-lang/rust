// Checks that rustc correctly errors when passed an invalid lint with
// `--force-warns`. This is a regression test for issue #86958.
//
// compile-flags: -Z unstable-options --force-warns foo-qux
// error-pattern: unknown lint: `foo_qux`

fn main() {}
