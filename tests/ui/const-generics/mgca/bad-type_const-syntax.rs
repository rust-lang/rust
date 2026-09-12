trait Tr {
    #[rustc_always_gca]
    //~^ ERROR: the `rustc_always_gca` attribute is an experimental feature [E0658]
    const N: usize;
}

struct S;

impl Tr for S {
    const N: usize = core::direct_const_arg!(0);
    //~^ ERROR: use of unstable library feature `min_generic_const_args` [E0658]
    //~| ERROR: implementation of a `#[rustc_always_gca]` must have a `direct_const_arg!` RHS
    //~| ERROR: expected expression, found `direct_const_arg!()` constant
}

fn main() {}
