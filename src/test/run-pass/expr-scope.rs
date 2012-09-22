// Regression test for issue #762
// xfail-fast

#[legacy_exports];

fn f() { }
fn main() { return ::f(); }
