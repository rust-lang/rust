//! Regression test for <https://github.com/rust-lang/rust/issues/144215>

#[rustfmt::skip]
struct S<T:>(&'static T);
//~^ ERROR the parameter type `T` may not live long enough
//~| HELP consider adding an explicit lifetime bound
//~| SUGGESTION  'static

fn main() {}
