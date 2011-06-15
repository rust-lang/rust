


// -*- rust -*-
fn main() { assert (even(42)); assert (odd(45)); }

fn even(int n) -> bool { if (n == 0) { ret true; } else { be odd(n - 1); } }

fn odd(int n) -> bool { if (n == 0) { ret false; } else { be even(n - 1); } }