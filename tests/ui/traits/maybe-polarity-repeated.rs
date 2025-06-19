#![feature(more_maybe_bounds)]

trait Trait {}
fn foo<T: ?Trait + ?Trait>(_: T) {}
//~^ ERROR type parameter has more than one relaxed default bound, only one is supported
//~| ERROR bound modifier `?` can only be applied to default traits like `Sized`
//~| ERROR bound modifier `?` can only be applied to default traits like `Sized`

fn main() {}
