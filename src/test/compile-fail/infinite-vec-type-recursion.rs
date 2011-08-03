// -*- rust -*-
// error-pattern: illegal recursive type

type x = vec[x];

fn main() { let b: x = []; }