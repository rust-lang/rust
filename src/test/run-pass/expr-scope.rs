// Regression test for issue #762
// xfail-fast

fn f() { }
fn main() { return ::f(); }
