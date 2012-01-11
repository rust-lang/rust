


// -*- rust -*-
fn foo(f: native fn(int) -> int) { }

fn id(x: int) -> int { ret x; }

fn main() { foo(id); }
