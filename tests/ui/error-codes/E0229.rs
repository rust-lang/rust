pub trait Foo {
    type A;
    fn boo(&self) -> <Self as Foo>::A;
}

struct Bar;

impl Foo for isize {
    type A = usize;
    fn boo(&self) -> usize { 42 }
}

fn baz<I>(x: &<I as Foo<A = Bar>>::A) {}
//~^ ERROR associated item constraints are not allowed here [E0229]
//~| ERROR associated item constraints are not allowed here [E0229]
//~| ERROR associated item constraints are not allowed here [E0229]
//~| ERROR the trait bound `I: Foo` is not satisfied
//~| ERROR the trait bound `I: Foo` is not satisfied

fn main() {
}
