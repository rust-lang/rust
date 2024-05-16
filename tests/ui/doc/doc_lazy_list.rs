#![warn(clippy::doc_lazy_continuation)]

/// 1. nest here
/// lazy continuation
//~^ ERROR: doc list item missing indentation
fn one() {}

/// 1. first line
/// lazy list continuations don't make warnings with this lint
//~^ ERROR: doc list item missing indentation
/// because they don't have the
//~^ ERROR: doc list item missing indentation
fn two() {}

///   - nest here
/// lazy continuation
//~^ ERROR: doc list item missing indentation
fn three() {}

///   - first line
/// lazy list continuations don't make warnings with this lint
//~^ ERROR: doc list item missing indentation
/// because they don't have the
//~^ ERROR: doc list item missing indentation
fn four() {}

///   - nest here
/// lazy continuation
//~^ ERROR: doc list item missing indentation
fn five() {}

///   - - first line
/// this will warn on the lazy continuation
//~^ ERROR: doc list item missing indentation
///     and so should this
//~^ ERROR: doc list item missing indentation
fn six() {}

///   - - first line
///
///     this is not a lazy continuation
fn seven() {}
