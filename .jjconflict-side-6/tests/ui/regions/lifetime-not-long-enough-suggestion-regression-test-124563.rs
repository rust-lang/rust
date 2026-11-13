// #124563
use std::marker::PhantomData;

pub trait Trait {}

pub trait Foo {
    type Trait: Trait;
    type Bar: Bar;
    fn foo(&mut self);
}

pub struct FooImpl<'a, 'b, A: Trait>(PhantomData<&'a &'b A>);

impl<'a, 'b, T> Foo for FooImpl<'a, 'b, T>
where
    T: Trait,
{
    type Trait = T;
    type Bar = BarImpl<'a, 'b, T>; //~ ERROR lifetime bound not satisfied

    fn foo(&mut self) {
        self.enter_scope(|ctx| { //~ ERROR lifetime may not live long enough
            BarImpl(ctx); //~ ERROR lifetime may not live long enough
        });
    }
}

impl<'a, 'b, T> FooImpl<'a, 'b, T>
where
    T: Trait,
{
    fn enter_scope(&mut self, _scope: impl FnOnce(&mut Self)) {}
}
pub trait Bar {
    type Foo: Foo;
}

pub struct BarImpl<'a, 'b, T: Trait>(&'b mut FooImpl<'a, 'b, T>);

impl<'a, 'b, T> Bar for BarImpl<'a, 'b, T>
where
    T: Trait,
{
    type Foo = FooImpl<'a, 'b, T>;
}

fn main() {}
