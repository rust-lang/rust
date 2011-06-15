


// -*- rust -*-
fn f(int n) -> int { ret n; }

fn main() {
    let fn() -> int  g = bind f(10);
    let int i = g();
    assert (i == 10);
}