// xfail-test
// -*- rust -*-

// error-pattern: enum of infinite size

enum mlist { cons(int, mlist); nil; }

fn main() { let a = cons(10, cons(11, nil)); }