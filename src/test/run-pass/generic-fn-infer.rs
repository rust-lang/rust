


// -*- rust -*-

// Issue #45: infer type parameters in function applications
fn id<copy T>(x: T) -> T { ret x; }

fn main() { let x: int = 42; let y: int = id(x); assert (x == y); }
