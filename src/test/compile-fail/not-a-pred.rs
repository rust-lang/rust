// -*- rust -*-
// xfail-stage0
// error-pattern: Non-predicate in constraint: lt

fn f(a: int, b: int) { }

obj lt(a: int, b: int) { }

fn main() { let a: int = 10; let b: int = 23; check (lt(a, b)); f(a, b); }