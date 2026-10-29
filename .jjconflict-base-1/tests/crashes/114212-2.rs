//@ known-bug: #114212
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

const SOME_CONST: usize = 1;

struct UwU<
    // have a const generic with a default that's from another const item
    // (associated consts work, a const declared in a block here, inline_const, etc)
    const N: usize = SOME_CONST,
    // use the previous const in a type generic
    A = [(); N],
> {
    // here to suppress "unused generic" error if the code stops ICEing
    _x: core::marker::PhantomData<A>,
}
