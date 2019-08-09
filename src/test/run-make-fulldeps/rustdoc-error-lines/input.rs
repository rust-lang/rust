// Test for #45868

// random #![feature] to ensure that crate attrs
// do not offset things
/// ```rust
/// #![feature(nll)]
/// let x: char = 1;
/// ```
pub fn foo() {

}

/// Add some text around the test...
///
/// ```rust
/// #![feature(nll)]
/// let x: char = 1;
/// ```
///
/// ...to make sure that the line number is still correct.
///
/// Let's also add a second test in the same doc comment.
///
/// ```rust
/// #![feature(nll)]
/// let x: char = 1;
/// ```
pub fn bar() {}
