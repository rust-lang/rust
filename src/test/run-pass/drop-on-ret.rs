


// -*- rust -*-
fn f() -> int { if true { let s: ~str = ~"should not leak"; return 1; } return 0; }

fn main() { f(); }
