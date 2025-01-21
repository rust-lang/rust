#![feature(auto_traits)]

trait Trait1 {}
auto trait Trait2 {}
trait Trait3: ?Trait1 {}
//~^  ERROR `?Trait` is not permitted in supertraits
trait Trait4 where Self: ?Trait1 {}
//~^ ERROR ?Trait` bounds are only permitted at the point where a type parameter is declared

fn foo(_: Box<dyn Trait1 + ?Trait2>) {}
//~^  ERROR `?Trait` is not permitted in trait object types
fn bar<T: ?Trait1 + ?Trait2>(_: T) {}
//~^ ERROR type parameter has more than one relaxed default bound, only one is supported
//~| ERROR relaxing a default bound only does something for `?Sized`; all other traits are not bound by default
//~| ERROR relaxing a default bound only does something for `?Sized`; all other traits are not bound by default

trait Trait {}
// Do not suggest `#![feature(more_maybe_bounds)]` for repetitions
fn baz<T: ?Trait + ?Trait>(_ : T) {}
//~^ ERROR type parameter has more than one relaxed default bound, only one is supported
//~| ERROR relaxing a default bound only does something for `?Sized`; all other traits are not bound by default
//~| ERROR relaxing a default bound only does something for `?Sized`; all other traits are not bound by default

fn main() {}
