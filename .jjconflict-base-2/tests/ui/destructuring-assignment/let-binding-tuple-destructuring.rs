//! Checks basic multiple variable declaration using tuple destructuring in a `let` binding.

//@ run-pass

pub fn main() {
    let (x, y) = (10, 20);
    let z = x + y;
    assert_eq!(z, 30);
}
