#![feature(min_generic_const_args)]
#![expect(incomplete_features)]
use std::gca;
trait Trait {
    #[rustc_always_gca]
    const N: usize = gca!(10);
    //~^ ERROR associated type defaults are unstable
}

fn main() {}
