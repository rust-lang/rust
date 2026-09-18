const FOO: u8 = core::direct_const_arg!(10);
//~^ ERROR use of unstable library feature `min_generic_const_args` [E0658]
//~| ERROR expected expression, found `direct_const_arg!()` constant

trait Bar {
    #[rustc_always_gca]
    //~^ ERROR the `rustc_always_gca` attribute is an experimental feature [E0658]
    const BAR: bool;
}

impl Bar for bool {
    const BAR: bool = core::direct_const_arg!(false);
    //~^ ERROR use of unstable library feature `min_generic_const_args` [E0658]
    //~| ERROR expected expression, found `direct_const_arg!()` constant
    //~| ERROR implementation of a `#[rustc_always_gca]` must have a `direct_const_arg!` RHS
}

fn main() {}
