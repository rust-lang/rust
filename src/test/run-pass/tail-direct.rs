


// -*- rust -*-
fn main() { assert (even(42)); assert (odd(45)); }

fn even(n: int) -> bool { if n == 0 { ret true; } else { ret odd(n - 1); } }

fn odd(n: int) -> bool { if n == 0 { ret false; } else { ret even(n - 1); } }
