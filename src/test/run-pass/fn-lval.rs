


// -*- rust -*-
fn foo(fn(int) -> int  f) { }

fn id(int x) -> int { ret x; }

fn main() { foo(id); }