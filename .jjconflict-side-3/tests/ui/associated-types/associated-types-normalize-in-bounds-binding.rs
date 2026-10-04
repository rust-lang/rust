//@ run-pass
#![allow(unused_variables)]
// Test that we normalize associated types that appear in a bound that
// contains a binding. Issue #21664.


#![allow(dead_code)]

pub trait Integral {
    type Opposite;
}

impl Integral for i32 {
    type Opposite = u32;
}

impl Integral for u32 {
    type Opposite = i32;
}

pub trait FnLike<A> {
    type R;

    fn dummy(&self, a: A) -> Self::R { loop { } }
}

fn foo<T>()
    where T : FnLike<<i32 as Integral>::Opposite, R=bool>
{
    bar::<T>();
}

fn bar<T>()
    where T : FnLike<u32, R=bool>
{}

fn main() { }
