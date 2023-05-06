#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

trait Indices<const N:usize> {
    const NUM_ELEMS: usize = 0;
}

trait Concat {
    type Output;
}

struct Tensor<A: Indices<42>>(A)
where
    [(); A::NUM_ELEMS]: Sized;

impl<I: Indices<42>> Concat for Tensor<I>
where
    [(); I::NUM_ELEMS]: Sized
{
    type Output = Tensor<<I as Concat>::Output>;
    //~^ ERROR the trait bound `I: Concat` is not satisfied
}

fn main() {}
