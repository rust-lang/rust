//! This test verifies that the type checker correctly identifies and reports error

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
