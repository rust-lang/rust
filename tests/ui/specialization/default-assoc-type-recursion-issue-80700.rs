//@ check-pass

#![feature(specialization)]
#![allow(incomplete_features)]

// Tests that a blanket impl supplying a `default type` does not make a
// recursive trait requirement diverge.
// Regression test for #80700.

use std::marker::PhantomData;

struct Nil;
struct Cons<Head, Tail>(PhantomData<(Head, Tail)>);
struct Error;

trait GetLast {
    type Output;
}

impl<T> GetLast for T {
    default type Output = Error;
}

impl<Head> GetLast for Cons<Head, Nil> {
    type Output = Nil;
}

impl<Head, Head2, Tail2> GetLast for Cons<Head, Cons<Head2, Tail2>>
where
    Cons<Head2, Tail2>: GetLast,
{
    type Output = <Cons<Head2, Tail2> as GetLast>::Output;
}

fn main() {}
