


// -*- rust -*-
fn f(n: int) -> int { ret n; }

fn main() {
    let g: fn() -> int  = bind f(10);
    let i: int = g();
    assert (i == 10);
}