/// This is a test for needless `fn main()` in doctests.
///
/// # Examples
///
/// This should lint
/// ```
/// fn main() {
///     unimplemented!();
/// }
/// ```
fn bad_doctest() {}

/// # Examples
///
/// This shouldn't lint, because the `main` is empty:
/// ```
/// fn main(){}
/// ```
///
/// This shouldn't lint either, because there's a `static`:
/// ```
/// static ANSWER: i32 = 42;
///
/// fn main() {
///     assert_eq!(42, ANSWER);
/// }
/// ```
fn no_false_positives() {}

fn main() {
    bad_doctest();
    no_false_positives();
}
