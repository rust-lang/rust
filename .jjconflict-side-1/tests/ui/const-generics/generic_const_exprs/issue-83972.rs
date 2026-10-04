//@ build-pass

#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

pub trait Foo {
    fn foo(&self);
}

pub struct FooImpl<const N: usize>;
impl<const N: usize> Foo for FooImpl<N> {
    fn foo(&self) {}
}

pub trait Bar: 'static {
    type Foo: Foo;
    fn get() -> &'static Self::Foo;
}

struct BarImpl;
impl Bar for BarImpl {
    type Foo = FooImpl<
        {
            { 4 }
        },
    >;
    fn get() -> &'static Self::Foo {
        &FooImpl
    }
}

pub fn boom<B: Bar>() {
    B::get().foo();
}

fn main() {
    boom::<BarImpl>();
}
