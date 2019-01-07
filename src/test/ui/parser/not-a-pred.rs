// error-pattern: lt

fn f(a: isize, b: isize) : lt(a, b) { }

fn lt(a: isize, b: isize) { }

fn main() { let a: isize = 10; let b: isize = 23; check (lt(a, b)); f(a, b); }
