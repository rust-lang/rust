// compile-flags:--test

// the `--test` harness creates modules with these textual names, but
// they should be inaccessible from normal code.
use __test as x; //~ ERROR unresolved import `__test`
use __test_reexports as y; //~ ERROR unresolved import `__test_reexports`

#[test]
fn baz() {}
