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
/// With an explicit return type it should lint too
/// ```edition2015
/// fn main() -> () {
///     unimplemented!();
/// }
/// ```
///
/// This should, too.
/// ```rust
/// fn main() {
///     unimplemented!();
/// }
/// ```
///
/// This one too.
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
/// This shouldn't lint either, because main is async:
/// ```edition2018
/// async fn main() {
///     assert_eq!(42, ANSWER);
/// }
/// ```
///
/// Same here, because the return type is not the unit type:
/// ```
/// fn main() -> Result<()> {
///     Ok(())
/// }
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
/// This shouldn't lint either, because there's a `const`:
/// ```
/// fn main() {
///     assert_eq!(42, ANSWER);
/// }
///
/// const ANSWER: i32 = 42;
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
/// Neither should this lint because it has an extern block:
/// ```
/// extern {}
/// fn main() {
///     unimplemented!();
/// }
/// ```
///
/// This should not lint because there is another function defined:
/// ```
/// fn fun() {}
///
/// fn main() {
///     unimplemented!();
/// }
/// ```
///
/// We should not lint inside raw strings ...
/// ```
/// let string = r#"
/// fn main() {
///     unimplemented!();
/// }
/// "#;
/// ```
///
/// ... or comments
/// ```
/// // fn main() {
/// //     let _inception = 42;
/// // }
/// let _inception = 42;
/// ```
///
/// We should not lint ignored examples:
/// ```rust,ignore
/// fn main() {
///     unimplemented!();
/// }
/// ```
///
/// Or even non-rust examples:
/// ```text
/// fn main() {
///     is what starts the program
/// }
/// ```
fn no_false_positives() {}

/// Yields a parse error when interpreted as rust code:
/// ```
/// r#"hi"
/// ```
fn issue_6022() {}

fn main() {
    bad_doctests();
    no_false_positives();
}
