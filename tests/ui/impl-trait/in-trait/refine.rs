#![deny(refining_impl_trait)]

pub trait Foo {
    fn bar() -> impl Sized;
}

pub struct A;
impl Foo for A {
    fn bar() -> impl Copy {}
    //~^ ERROR impl method signature does not match trait method signature
}

pub struct B;
impl Foo for B {
    fn bar() {}
    //~^ ERROR impl method signature does not match trait method signature
}

pub struct C;
impl Foo for C {
    fn bar() -> () {}
    //~^ ERROR impl method signature does not match trait method signature
}

struct Private;
impl Foo for Private {
    fn bar() -> () {}
    //~^ ERROR impl method signature does not match trait method signature
}

pub trait Arg<A> {
    fn bar() -> impl Sized;
}
impl Arg<Private> for A {
    fn bar() -> () {}
    //~^ ERROR impl method signature does not match trait method signature
}

pub trait Late {
    fn bar<'a>(&'a self) -> impl Sized + 'a;
}

pub struct D;
impl Late for D {
    fn bar(&self) -> impl Copy + '_ {}
    //~^ ERROR impl method signature does not match trait method signature
}

mod unreachable {
    pub trait UnreachablePub {
        fn bar() -> impl Sized;
    }

    struct E;
    impl UnreachablePub for E {
        fn bar() {}
        //~^ ERROR impl method signature does not match trait method signature
    }
}

fn main() {}
