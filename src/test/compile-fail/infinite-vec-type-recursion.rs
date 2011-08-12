// -*- rust -*-
// error-pattern: illegal recursive type

type x = [x];

fn main() { let b: x = ~[]; }