


// -*- rust -*-
fn f() -> int { ret 42; }

fn main() {
    let g: fn() -> int  = bind f();
    let i: int = g();
    assert (i == 42);
}