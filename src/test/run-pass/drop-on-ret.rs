


// -*- rust -*-
fn f() -> int { if true { let s: str = "should not leak"; ret 1; } ret 0; }

fn main() { f(); }