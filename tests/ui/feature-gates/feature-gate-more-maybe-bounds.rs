#![feature(auto_traits)]

trait Trait1 {}
auto trait Trait2 {}
trait Trait3: ?Trait1 {} //~ ERROR relaxed bounds are not permitted in supertrait bounds
trait Trait4 where Self: ?Trait1 {} //~ ERROR this relaxed bound is not permitted here

fn foo(_: Box<dyn Trait1 + ?Trait2>) {}
//~^  ERROR relaxed bounds are not permitted in trait object types
fn bar<T: ?Trait1 + ?Trait2>(_: T) {}
//~^ ERROR bound modifier `?` can only be applied to `Sized`
//~| ERROR bound modifier `?` can only be applied to `Sized`

fn main() {}
