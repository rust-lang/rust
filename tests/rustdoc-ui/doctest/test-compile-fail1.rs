//@ compile-flags:--test

/// ```
/// assert!(true)
/// ```
pub fn f() {}

pub fn f() {}
//~^ ERROR the name `f` is defined multiple times
