// -*- rust -*-
// error-pattern: Non-predicate in constraint: lt

fn f(a: int, b: int) : lt(a, b) { }

fn lt(a: int, b: int) { }

fn main() { let a: int = 10; let b: int = 23; check (lt(a, b)); f(a, b); }
