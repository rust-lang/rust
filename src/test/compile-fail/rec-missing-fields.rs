// -*- rust -*-

// error-pattern: mismatched types

// Issue #51.

type point = {x: int, y: int};

fn main() { let p: point = {x: 10}; log(debug, p.y); }
