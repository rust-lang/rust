//@ run-pass
// This test verifies the short-circuiting behavior of logical operators `||` and `&&`.
// It ensures that the right-hand expression is not evaluated when the left-hand
// expression is sufficient to determine the result.

fn would_panic_if_called(x: &mut isize) -> bool {
    *x += 1;
    assert!(false, "This function should never be called due to short-circuiting");
    false
}

fn main() {
    let x = 1 == 2 || 3 == 3;
    assert!(x);

    let mut y: isize = 10;
    println!("Result of short-circuit: {}", x || would_panic_if_called(&mut y));
    assert_eq!(y, 10, "y should remain 10 if short-circuiting works correctly");

    if true && x {
        assert!(true);
    } else {
        assert!(false, "This branch should not be reached");
    }
}
