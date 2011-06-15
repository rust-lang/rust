


// -*- rust -*-
// xfail-stage0
pred f(int q) -> bool { ret true; }

fn main() { auto x = 0; check (f(x)); }