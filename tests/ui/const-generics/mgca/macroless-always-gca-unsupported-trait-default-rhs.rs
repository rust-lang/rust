#![feature(gca_min_const_items, gca_macroless_items, associated_type_defaults)]

trait Trait {
    #[rustc_always_gca]
    const ASSOC: usize = 1 + 1;
    //~^ ERROR the expression of this associated constant was unable to be treated as a gca constant
}

fn main() {}
