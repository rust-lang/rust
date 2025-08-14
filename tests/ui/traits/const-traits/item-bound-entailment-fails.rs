//@ compile-flags: -Znext-solver
#![feature(const_trait_impl)]

#[const_trait] trait Foo {
    type Assoc<T>: [const] Bar
    where
        T: [const] Bar;
}

#[const_trait] trait Bar {}
struct N<T>(T);
impl<T> Bar for N<T> where T: Bar {}
struct C<T>(T);
impl<T> const Bar for C<T> where T: [const] Bar {}

impl const Foo for u32 {
    type Assoc<T> = N<T>
    //~^ ERROR the trait bound `N<T>: [const] Bar` is not satisfied
    where
        T: [const] Bar;
}

impl const Foo for i32 {
    type Assoc<T> = C<T>
    //~^ ERROR the trait bound `T: [const] Bar` is not satisfied
    where
        T: Bar;
}

fn main() {}
