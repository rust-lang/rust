// Checks that rustc correctly errors when passed an invalid lint with
// `--force-warn`. This is a regression test for issue #86958.
//
// compile-flags: --force-warn foo-qux
// error-pattern: unknown lint: `foo_qux`

fn main() {}
