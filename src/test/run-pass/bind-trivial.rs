


// -*- rust -*-
fn f(n: int) -> int { ret n; }

fn main() {
    let g: fn(int) -> int  = bind f(_);
    let i: int = g(42);
    assert (i == 42);
}