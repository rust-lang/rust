//! Tests basic `usize` functionality

//@ run-pass

pub fn main() {
    // Literal matches assignment type
    let a: usize = 42usize;
    // Literal cast
    let b: usize = 42 as usize;
    // Literal type inference from assignment type
    let c: usize = 42;
    // Assignment type inference from literal (and later comparison)
    let d = 42usize;
    // Function return value type inference
    let e = return_val();

    assert_eq!(a, b);
    assert_eq!(a, c);
    assert_eq!(a, d);
    assert_eq!(a, e);
}

fn return_val() -> usize {
    42
}
