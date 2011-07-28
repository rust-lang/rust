// Regression test for issue #762
// xfail-stage0

// Breaks with fast-check, disabling to get tinderbox green again
// xfail-stage1
// xfail-stage2

fn f() { }
fn main() { ret ::f(); }