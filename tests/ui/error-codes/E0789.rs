//@ compile-flags: --crate-type lib

#![feature(rustc_attrs)]
#![feature(staged_api)]
#![unstable(feature = "foo_module", reason = "...", issue = "123")]

#[rustc_allowed_through_unstable_modules = "use stable path instead"]
// #[stable(feature = "foo", since = "1.0")]
struct Foo;
//~^ ERROR `rustc_allowed_through_unstable_modules` attribute must be paired with a `stable` attribute
