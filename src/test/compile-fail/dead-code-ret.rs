// xfail-test
// -*- rust -*-

// error-pattern: dead

fn f(caller: str) { log caller; }

fn main() { ret f("main"); log "Paul is dead"; }

