


// -*- rust -*-

// error-pattern:non-exhaustive match failure
tag t { a; b; }

fn main() { let x = a; alt x { b { } } }
