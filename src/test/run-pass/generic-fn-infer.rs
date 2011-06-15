


// -*- rust -*-

// Issue #45: infer type parameters in function applications
fn id[T](&T x) -> T { ret x; }

fn main() { let int x = 42; let int y = id(x); assert (x == y); }