#![feature(staged_api)]
#![allow(internal_features)]
#![stable(feature = "unit_test", since = "1.0.0")]

#[unstable(feature = "step_trait", issue = "42168")]
pub trait Unstable {}

#[stable(feature = "unit_test", since = "1.0.0")]
fn foo<T: Unstable>(_: T) {}

#[stable(feature = "unit_test", since = "1.0.0")]
pub fn bar<T>(t: T) { //~ HELP consider restricting type parameter `T` with unstable trait `Unstable`
    foo(t) //~ ERROR E0277
}
#[stable(feature = "unit_test", since = "1.0.0")]
pub fn baz<T>(t: std::ops::Range<T>) { //~ HELP consider restricting type parameter `T` with unstable trait
    for _ in t {} //~ ERROR E0277
}
fn main() {}
