


// -*- rust -*-
fn foo(f: fn(int) -> int) { }

fn id(x: int) -> int { ret x; }

fn main() { foo(id); }
