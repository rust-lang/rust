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
///
/// This should, too.
///
/// ```rust
/// fn main() {
///     unimplemented!();
/// }
/// ```
///
/// This one too.
///
/// ```no_run
/// fn main() {
///     unimplemented!();
/// }
/// ```
fn bad_doctests() {}

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
///
/// Neither should this lint because of `extern crate`:
/// ```
/// #![feature(test)]
/// extern crate test;
/// fn main() {
///     assert_eq(1u8, test::black_box(1));
/// }
/// ```
///
/// We should not lint ignored examples:
///
/// ```rust,ignore
/// fn main() {
///     unimplemented!();
/// }
/// ```
///
/// Or even non-rust examples:
///
/// ```text
/// fn main() {
///     is what starts the program
/// }
/// ```
fn no_false_positives() {}

fn main() {
    bad_doctests();
    no_false_positives();
}
