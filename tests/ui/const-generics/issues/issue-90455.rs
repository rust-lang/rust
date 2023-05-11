#![feature(generic_const_exprs, adt_const_params)]
#![allow(incomplete_features)]

struct FieldElement<const N: &'static str> {
    n: [u64; num_limbs(N)],
    //~^ ERROR unconstrained generic constant
}
const fn num_limbs(_: &str) -> usize {
    0
}

fn main() {}
