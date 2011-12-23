// xfail-test
// -*- rust -*-

// error-pattern: dead

fn f(caller: str) { log(debug, caller); }

fn main() { ret f("main"); #debug("Paul is dead"); }

