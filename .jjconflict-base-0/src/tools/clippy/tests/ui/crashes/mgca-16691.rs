//@ check-pass
#![expect(incomplete_features)]
#![feature(gca_min_const_items)]

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
