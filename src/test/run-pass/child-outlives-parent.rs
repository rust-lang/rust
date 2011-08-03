// Reported as issue #126, child leaks the string.
fn child2(s: str) { }

fn main() { let x = spawn child2("hi"); }