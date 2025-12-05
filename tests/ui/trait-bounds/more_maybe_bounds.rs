#![feature(auto_traits, more_maybe_bounds, negative_impls)]

trait Trait1 {}
auto trait Trait2 {}

trait Trait3: ?Trait1 {}
//~^ ERROR bound modifier `?` can only be applied to default traits
//~| ERROR bound modifier `?` can only be applied to default traits
//~| ERROR bound modifier `?` can only be applied to default traits
trait Trait4 where Self: Trait1 {}


fn foo(_: Box<(dyn Trait3 + ?Trait2)>) {}
//~^ ERROR bound modifier `?` can only be applied to default traits

fn bar<T: ?Sized + ?Trait2 + ?Trait1 + ?Trait4>(_: &T) {}
//~^ ERROR bound modifier `?` can only be applied to default traits
//~| ERROR bound modifier `?` can only be applied to default traits
//~| ERROR bound modifier `?` can only be applied to default traits

fn baz<T>() where T: Iterator<Item: ?Trait1> {}
//~^ ERROR this relaxed bound is not permitted here
//~| ERROR bound modifier `?` can only be applied to default traits

struct S1<T>(T);

impl<T> S1<T> {
    fn f() where T: ?Trait1 {}
    //~^ ERROR this relaxed bound is not permitted here
    //~| ERROR bound modifier `?` can only be applied to default traits
}

trait Trait5<'a> {}

struct S2<T>(T) where for<'a> T: ?Trait5<'a>;
//~^ ERROR this relaxed bound is not permitted here
//~| ERROR bound modifier `?` can only be applied to default traits

struct S;
impl !Trait2 for S {}
impl Trait1 for S {}
impl Trait3 for S {}

fn main() {
    foo(Box::new(S));
    bar(&S);
}
