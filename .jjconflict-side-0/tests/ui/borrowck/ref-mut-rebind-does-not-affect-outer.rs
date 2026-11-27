//! Regression test for https://github.com/rust-lang/rust/issues/29053
//@ run-pass
fn main() {
    let x: &'static str = "x";

    {
        let y = "y".to_string();
        let ref mut x = &*x;
        *x = &*y;
    }

    assert_eq!(x, "x");
}
