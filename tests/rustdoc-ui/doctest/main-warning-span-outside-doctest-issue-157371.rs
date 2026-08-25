// Regression test for #157371.
// The warning for a top-level expression in a doctest should point at the stray
// semicolon inside the doctest, not at unrelated source below the doc comment.

//@ compile-flags:--test
//@ normalize-stdout: "tests/rustdoc-ui/doctest" -> "$$DIR"
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ check-pass

/// This part creates the doctest with a stray `;`.
///
/// ```
/// #[derive(Debug, PartialEq)]
/// struct Type {
///     left: i32,
///     right: i32,
/// };  // <- Stray `;`.
//~^ WARN the `main` function of this doctest won't be run
///
/// fn main() {
///     let x = Type {
///         left: 10,
///         right: 20,
///     };
///     assert_eq!(
///         x,
///         Type {
///             left: 10,
///             right: 20,
///         },
///     );
/// }
/// ```
pub fn add(left: u64, right: u64) -> u64 {
    // This code is completely unrelated to the doctest.
    left + right
}
