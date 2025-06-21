// FIXME(more_maybe_bounds): This used to be a check-pass test before rustc started to reject
// relaxed bounds of *non-default* traits. Therefore, the original test intention has been lost.
#![feature(auto_traits, more_maybe_bounds, negative_impls)]

trait Trait1 {}
auto trait Trait2 {}

trait Trait3 : ?Trait1 {}
trait Trait4 where Self: Trait1 {}

fn foo(_: Box<(dyn Trait3 + ?Trait2)>) {}
fn bar<T: ?Sized + ?Trait2 + ?Trait1 + ?Trait4>(_: &T) {}
//~^ ERROR bound modifier `?` can only be applied to default traits like `Sized`
//~| ERROR bound modifier `?` can only be applied to default traits like `Sized`
//~| ERROR bound modifier `?` can only be applied to default traits like `Sized`

struct S;
impl !Trait2 for S {}
impl Trait1 for S {}
impl Trait3 for S {}

fn main() {
    foo(Box::new(S));
    bar(&S);
}
