//@ revisions: cfail
#![feature(generic_const_exprs)]
#![allow(incomplete_features, unused_braces)]

struct Buffer<T, const S: usize>
where
    [(); { S * 2 }]: Default,
{
    data: [T; { S * 2 }],
}

struct BufferIter<'a, T, const S: usize>(&'a Buffer<T, S>)
where
    [(); { S * 2 }]: Default;

impl<'a, T, const S: usize> Iterator for BufferIter<'a, T, S> {
    //~^ error: the trait bound
    //~| error: unconstrained generic constant
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        //~^ error: the trait bound
        //~| error: unconstrained generic constant
        //~| error: the trait bound
        //~| error: unconstrained generic constant
        None
    }
}

fn main() {}
