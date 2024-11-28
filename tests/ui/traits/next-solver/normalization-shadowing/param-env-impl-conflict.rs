//@ compile-flags: -Znext-solver
//@ check-pass

// Regression test for #119608.

pub trait Foo {}

pub trait Bar {
    type Assoc;
}

impl<T: Foo> Bar for T {
    type Assoc = T;
}

pub fn foo<I>(_input: <I as Bar>::Assoc)
where
    I: Bar,
    <I as Bar>::Assoc: Foo,
{
}

fn main() {}
