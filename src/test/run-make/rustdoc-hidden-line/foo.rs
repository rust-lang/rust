#[crate_id="foo#0.1"];

/// The '# ' lines should be removed from the output, but the #[deriving] should be
/// retained.
///
/// ```rust
/// mod to_make_deriving_work { // FIXME #4913
///
/// # #[deriving(Eq)] // invisible
/// # struct Foo; // invisible
///
/// #[deriving(Eq)] // Bar
/// struct Bar(Foo);
///
/// fn test() {
///     let x = Bar(Foo);
///     assert!(x == x); // check that the derivings worked
/// }
///
/// }
/// ```
pub fn foo() {}
