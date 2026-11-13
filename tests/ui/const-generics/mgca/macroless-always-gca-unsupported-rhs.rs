#![feature(gca_min_const_items, gca_macroless_items)]

trait Trait {
    #[rustc_always_gca]
    const ASSOC: usize;
}

impl<const N: usize> Trait for [u8; N] {
    const ASSOC: usize = N + 1;
    //~^ ERROR this expression cannot be marked as direct
}

// "this expression cannot be marked as direct" error not shown when rhs does not exists
impl<const N: usize> Trait for [u16; N] {
    const ASSOC: usize;
    //~^ ERROR associated constant in `impl` without body
}

fn main() {}
