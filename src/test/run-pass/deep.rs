


// -*- rust -*-
fn f(x: int) -> int {
    if x == 1 { return 1; } else { let y: int = 1 + f(x - 1); return y; }
}

fn main() { assert (f(5000) == 5000); }
