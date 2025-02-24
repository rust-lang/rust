#![feature(auto_traits)]
#![feature(more_maybe_bounds)]
#![feature(negative_impls)]

trait Trait1 {}
auto trait Trait2 {}

trait Trait3 : ?Trait1 {}
trait Trait4 where Self: Trait1 {}

fn foo(_: Box<(dyn Trait3 + ?Trait2)>) {}
fn bar<T: ?Sized + ?Trait2 + ?Trait1 + ?Trait4>(_: &T) {}
//~^ ERROR relaxing a default bound only does something for `?Sized`; all other traits are not bound by default
//~| ERROR relaxing a default bound only does something for `?Sized`; all other traits are not bound by default
//~| ERROR relaxing a default bound only does something for `?Sized`; all other traits are not bound by default

struct S;
impl !Trait2 for S {}
impl Trait1 for S {}
impl Trait3 for S {}

fn main() {
    foo(Box::new(S));
    bar(&S);
}
