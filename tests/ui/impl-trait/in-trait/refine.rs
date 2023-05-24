#![feature(return_position_impl_trait_in_trait, async_fn_in_trait)]

trait Foo {
    fn bar() -> impl Sized;
}

struct A;
impl Foo for A {
    fn bar() -> impl Copy {}
    //~^ ERROR impl method signature does not match trait method signature
}

struct B;
impl Foo for B {
    fn bar() {}
    //~^ ERROR impl method signature does not match trait method signature
}

struct C;
impl Foo for C {
    fn bar() -> () {}
    //~^ ERROR impl method signature does not match trait method signature
}

trait Late {
    fn bar<'a>(&'a self) -> impl Sized + 'a;
}

struct D;
impl Late for D {
    fn bar(&self) -> impl Copy + '_ {}
    //~^ ERROR impl method signature does not match trait method signature
}

trait Captures<'a, 'b> {}
impl<T> Captures<'_, '_> for T {}
trait Outlives {
    fn bar<'a, 'b>() -> impl Captures<'a, 'b>;
}

struct E;
impl Outlives for E {
    fn bar<'a, 'b>() -> impl Captures<'a, 'b> + 'b {}
    //~^ ERROR impl method signature does not match trait method signature
}

fn main() {}
