// Ensure that we actually enforce equality constraints found in trait object types.

#![feature(gca_min_const_items)]
#![expect(incomplete_features)]

use std::gca;

trait Trait {
    #[rustc_always_gca]
    const N: usize;
}

impl Trait for () {
    const N: usize = gca!(1);
}

fn main() {
    let _: &dyn Trait<N = 0> = &(); //~ ERROR type mismatch resolving `<() as Trait>::N == 0`
}
