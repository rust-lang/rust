


// -*- rust -*-
fn f(int n) -> int { ret n; }

fn main() {
    let fn(int) -> int  g = bind f(_);
    let int i = g(42);
    assert (i == 42);
}