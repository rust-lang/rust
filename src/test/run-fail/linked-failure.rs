// -*- rust -*-

// error-pattern:1 == 2
// no-valgrind

fn child() { assert (1 == 2); }

fn main() { let p: port[int] = port(); spawn child(); let x: int; p |> x; }