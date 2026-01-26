// Ensure that we actually enforce equality constraints found in trait object types.

#![feature(min_generic_const_args)]
#![expect(incomplete_features)]

trait Trait {
    #[type_const]
    const N: usize;
}

impl Trait for () {
    #[type_const]
    const N: usize = 1;
}

fn main() {
    let _: &dyn Trait<N = 0> = &(); //~ ERROR type mismatch resolving `<() as Trait>::N == 0`
}
