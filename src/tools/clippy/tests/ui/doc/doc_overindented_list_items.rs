#![warn(clippy::doc_overindented_list_items)]

#[rustfmt::skip]
/// - first list item
///        overindented line
//~^ ERROR: doc list item overindented
///      this is overindented line too
//~^ ERROR: doc list item overindented
/// - second list item
fn foo() {}

#[rustfmt::skip]
///   - first list item
///        overindented line
//~^ ERROR: doc list item overindented
///      this is overindented line too
//~^ ERROR: doc list item overindented
///   - second list item
fn bar() {}

#[rustfmt::skip]
/// * first list item
///        overindented line
//~^ ERROR: doc list item overindented
///      this is overindented line too
//~^ ERROR: doc list item overindented
/// * second list item
fn baz() {}
