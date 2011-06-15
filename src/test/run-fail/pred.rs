


// -*- rust -*-
// xfail-stage0
// error-pattern:Predicate lt(b, a) failed
fn f(int a, int b) { }

pred lt(int a, int b) -> bool { ret a < b; }

fn main() { let int a = 10; let int b = 23; check (lt(b, a)); f(b, a); }