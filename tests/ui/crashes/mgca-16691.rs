//@ check-pass
#![allow(incomplete_features)]
#![feature(min_generic_const_args)]

trait Trait {
    type const N: usize;
    fn process();
}

impl Trait for () {
    type const N: usize = 3;
    fn process() {
        const N: usize = <()>::N;
        _ = 0..Self::N;
    }
}
