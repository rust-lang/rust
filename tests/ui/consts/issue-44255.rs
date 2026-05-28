//@ run-pass

use std::marker::PhantomData;

fn main() {
    let _arr = [1; <Multiply<Five, Five>>::VAL];
}

trait TypeVal<T> {
    const VAL: T;
}

struct Five;

impl TypeVal<usize> for Five {
    const VAL: usize = 5;
}

struct Multiply<N, M> {
    _n: PhantomData<N>,
    _m: PhantomData<M>,
}

impl<N, M> TypeVal<usize> for Multiply<N, M>
    where N: TypeVal<usize>,
          M: TypeVal<usize>,
{
    const VAL: usize = N::VAL * M::VAL;
}
