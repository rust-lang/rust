//! Test basic closure syntax and usage with generic functions.
//!
//! This test checks that closure syntax works correctly for:
//! - Closures with parameters and return values
//! - Closures without parameters (both expression and block forms)
//! - Integration with generic functions and FnOnce trait bounds

//@ run-pass

fn f<F>(i: isize, f: F) -> isize
where
    F: FnOnce(isize) -> isize,
{
    f(i)
}

fn g<G>(_g: G)
where
    G: FnOnce(),
{
}

pub fn main() {
    // Closure with parameter that returns the same value
    assert_eq!(f(10, |a| a), 10);

    // Closure without parameters - expression form
    g(|| ());

    // Test closure reuse in generic context
    assert_eq!(f(10, |a| a), 10);

    // Closure without parameters - block form
    g(|| {});
}
