// Ensure that we actually enforce equality constraints found in trait object types.

#![feature(min_generic_const_args)]
#![expect(incomplete_features)]

trait Trait {
    #[rustc_always_gca]
    const N: usize;
}

impl Trait for () {
    const N: usize = core::direct_const_arg!(1);
}

fn main() {
    let _: &dyn Trait<N = 0> = &(); //~ ERROR type mismatch resolving `<() as Trait>::N == 0`
}
