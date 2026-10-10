//@ check-pass
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

struct Num<const N: usize>;

trait NumT {
    const VALUE: usize;
}

impl<const N: usize> NumT for Num<N> {
    const VALUE: usize = N;
}

struct Foo<'a, N: NumT>(&'a [u32; N::VALUE]) where [(); N::VALUE]:;

trait Bar {
    type Size: NumT;

    fn bar<'a>(foo: &Foo<'a, Self::Size>) where [(); Self::Size::VALUE]: {
        todo!();
    }
}

trait Baz<'a> {
    type Size: NumT;

    fn baz(foo: &Foo<'a, Self::Size>) where [(); Self::Size::VALUE]: {
        todo!();
    }
}

fn main() {}
