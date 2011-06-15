


// -*- rust -*-

// error-pattern:non-exhaustive match failure
tag t { a; b; }

fn main() { auto x = a; alt (x) { case (b) { } } }