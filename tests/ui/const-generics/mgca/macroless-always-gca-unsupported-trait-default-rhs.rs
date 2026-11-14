#![feature(gca_min_const_items, gca_macroless_items, associated_type_defaults)]

trait Trait {
    #[rustc_always_gca]
    const ASSOC: usize = 1 + 1;
    //~^ ERROR this expression cannot be marked as direct
}

fn main() {}
