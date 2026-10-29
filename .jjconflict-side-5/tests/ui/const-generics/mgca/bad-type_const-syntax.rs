trait Tr {
    #[rustc_always_gca]
    //~^ ERROR: the `rustc_always_gca` attribute is an experimental feature [E0658]
    const N: usize;
    #[rustc_always_gca]
    //~^ ERROR: the `rustc_always_gca` attribute is an experimental feature [E0658]
    const M: usize;
}

struct S;

impl Tr for S {
    const N: usize = core::gca!(0);
    //~^ ERROR: use of unstable library feature `gca_min_const_items` [E0658]
    //~| ERROR: implementation of a `#[rustc_always_gca]` must have a `gca!` RHS
    //~| ERROR: expected expression, found `gca!()` constant
    const M: usize = std::gca!(0);
    //~^ ERROR: use of unstable library feature `gca_min_const_items` [E0658]
    //~| ERROR: implementation of a `#[rustc_always_gca]` must have a `gca!` RHS
    //~| ERROR: expected expression, found `gca!()` constant
}

fn main() {}
