


// -*- rust -*-
fn foo(f: extern fn(int) -> int) { }

fn id(x: int) -> int { ret x; }

fn main() { foo(id); }
