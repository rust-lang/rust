//! Tests basic `isize` functionality

//@ run-pass

pub fn main() {
    // Literal matches assignment type
    let a: isize = 42isize;
    // Literal cast
    let b: isize = 42 as isize;
    // Literal type inference from assignment type
    let c: isize = 42;
    // Assignment type inference from literal (and later comparison)
    let d = 42isize;
    // Function return value type inference
    let e = return_val();

    assert_eq!(a, b);
    assert_eq!(a, c);
    assert_eq!(a, d);
    assert_eq!(a, e);
}

fn return_val() -> isize {
    42
}
