// xfail-test
// -*- rust -*-

// error-pattern: dead

fn f(caller: str) { log caller; }

fn main() { be f("main"); log "Paul is dead"; }

