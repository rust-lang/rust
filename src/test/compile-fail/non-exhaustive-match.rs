// -*- rust -*-
// error-pattern: Non-exhaustive pattern
enum t { a, b, }

fn main() { let x = a; alt x { b { } } }
