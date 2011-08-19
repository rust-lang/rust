// -*- rust -*-
// error-pattern:Predicate lt(b, a) failed
fn f(a: int, b: int) { }

pred lt(a: int, b: int) -> bool { ret a < b; }

fn main() { let a: int = 10; let b: int = 23; check (lt(b, a)); f(b, a); }
