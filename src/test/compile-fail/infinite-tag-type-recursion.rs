// xfail-test
// -*- rust -*-

// error-pattern: tag of infinite size

tag mlist { cons(int, mlist); nil; }

fn main() { let a = cons(10, cons(11, nil)); }