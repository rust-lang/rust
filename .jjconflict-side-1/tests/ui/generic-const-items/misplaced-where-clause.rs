//@ run-rustfix

#![feature(generic_const_items)]
#![allow(incomplete_features, dead_code)]

const K<T>: u64
where
    T: Tr<()>
= T::K;
//~^^^ ERROR where clauses are not allowed before const item bodies

trait Tr<P> {
    const K: u64
    where
        P: Copy
    = 0;
    //~^^^ ERROR where clauses are not allowed before const item bodies
}

fn main() {}
