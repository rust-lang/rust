//! Test for type mismatch error when returning `usize` from `isize` function.

fn f() -> isize {
    return g();
    //~^ ERROR mismatched types [E0308]
}

fn g() -> usize {
    return 0;
}

fn main() {
    let y = f();
}
