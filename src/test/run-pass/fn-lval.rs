


// -*- rust -*-
fn foo(f: extern fn(int) -> int) { }

fn id(x: int) -> int { return x; }

fn main() { foo(id); }
