#![feature(more_maybe_bounds)]

trait Trait {}
fn foo<T: ?Trait + ?Trait>(_: T) {}
//~^ ERROR type parameter has more than one relaxed default bound, only one is supported
//~| ERROR relaxing a default bound only does something for `?Sized`; all other traits are not bound by default
//~| ERROR relaxing a default bound only does something for `?Sized`; all other traits are not bound by default

fn main() {}
