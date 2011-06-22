


// xfail-stage0
// Reported as issue #126, child leaks the string.
fn child2(str s) { }

fn main() { auto x = spawn child2("hi"); }