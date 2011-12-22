// xfail-test
// -*- rust -*-

// error-pattern: dead

fn f(caller: str) { log_full(core::debug, caller); }

fn main() { be f("main"); #debug("Paul is dead"); }

