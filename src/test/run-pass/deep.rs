


// -*- rust -*-
fn f(int x) -> int {
    if (x == 1) { ret 1; } else { let int y = 1 + f(x - 1); ret y; }
}

fn main() { assert (f(5000) == 5000); }