// Test for #45868

// random #![feature] to ensure that crate attrs
// do not offset things
/// ```rust
/// #![feature(nll)]
/// let x: char = 1;
/// ```
pub fn foo() {

}
