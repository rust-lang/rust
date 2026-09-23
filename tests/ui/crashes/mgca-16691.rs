//@ check-pass
#![expect(incomplete_features)]
#![feature(min_generic_const_args)]

use std::gca;

trait Trait {
    #[rustc_always_gca]
    const N: usize;
    fn process();
}

impl Trait for () {
    const N: usize = gca!(3);
    fn process() {
        const N: usize = <()>::N;
        _ = 0..Self::N;
    }
}
