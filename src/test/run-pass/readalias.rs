


// -*- rust -*-
type point = {x: int, y: int, z: int};

fn f(p: &point) { assert (p.z == 12); }

fn main() { let x: point = {x: 10, y: 11, z: 12}; f(x); }