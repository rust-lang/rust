//@ compile-flags:--test

// the `--test` harness creates modules with these textual names, but
// they should be inaccessible from normal code.
use main as x; //~ ERROR unresolved import `main`
use test as y; //~ ERROR unresolved import `test`

#[test]
fn baz() {}
