//@ compile-flags: -Znext-solver
//@ check-pass

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

impl Foo for u32 {
    type Assoc<T> = N<T>
    where
        T: Bar;
}

impl const Foo for i32 {
    type Assoc<T> = C<T>
    where
        T: [const] Bar;
}

fn main() {}
