#![feature(min_generic_const_args)]
#![expect(incomplete_features)]
trait Trait {
    #[rustc_always_gca]
    const N: usize = core::direct_const_arg!(10);
    //~^ ERROR associated type defaults are unstable
}

fn main() {}
