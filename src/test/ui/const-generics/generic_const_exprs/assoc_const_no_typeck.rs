// check-pass
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

trait Trait<const N: usize> {
    const ASSOC: usize;
    type Foo;
}

fn no_cycle<const N: usize>()
where
    u8: Trait<N>,
    (): Trait<{ <u8 as Trait<N>>::ASSOC }>,
    [(); <() as Trait<{ <u8 as Trait<N>>::ASSOC }>>::ASSOC]: ,
{
}

fn foo<const N: usize>(_: [(); <<() as Trait<N>>::Foo as Trait<N>>::ASSOC])
where
    (): Trait<N>,
    <() as Trait<N>>::Foo: Trait<N>,
{
}

trait Trait2<T> {
    type Foo;
}

struct Inherent;
impl Inherent {
    const ASSOC: usize = 10;
}

fn bar<T>()
where
    (): Trait2<T, Foo = Inherent>,
    [(); <() as Trait2<T>>::Foo::ASSOC]: ,
{
}

fn main() {}
