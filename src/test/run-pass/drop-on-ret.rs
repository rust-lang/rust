


// -*- rust -*-
fn f() -> int { if (true) { let str s = "should not leak"; ret 1; } ret 0; }

fn main() { f(); }