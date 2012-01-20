


// -*- rust -*-

// error-pattern:non-exhaustive match failure
enum t { a; b; }

fn main() { let x = a; alt x { b { } } }
