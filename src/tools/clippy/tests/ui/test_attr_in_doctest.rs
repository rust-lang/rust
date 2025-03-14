/// This is a test for `#[test]` in doctests
///
/// # Examples
///
/// ```
/// #[test]
//~^ test_attr_in_doctest
/// fn should_be_linted() {
///     assert_eq!(1, 1);
/// }
/// ```
/// 
/// Make sure we catch multiple tests in one example,
/// and show that we really parse the attr:
/// ```
/// #[test]
//~^ test_attr_in_doctest
/// fn should_also_be_linted() {
///     #[cfg(test)]
///     assert!(true);
/// }
///
/// #[test]
//~^ test_attr_in_doctest
/// fn should_be_linted_too() {
///     assert_eq!("#[test]", "
///     #[test]
///     ");
/// }
/// ```
/// 
/// We don't catch examples that aren't run:
/// ```ignore
/// #[test]
/// fn ignored() { todo!() }
/// ```
/// ```no_run
/// #[test]
/// fn ignored() { todo!() }
/// ```
/// ```compile_fail
/// #[test]
/// fn ignored() { Err(()) }
/// ```
/// ```txt
/// #[test]
/// fn not_even_rust() { panic!("Ouch") }
/// ```
fn test_attr_in_doctests() {}
