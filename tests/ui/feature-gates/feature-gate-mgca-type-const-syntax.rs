const FOO: u8 = std::gca!(10);
//~^ ERROR use of unstable library feature `min_generic_const_args` [E0658]
//~| ERROR expected expression, found `gca!()` constant

trait Bar {
    #[rustc_always_gca]
    //~^ ERROR the `rustc_always_gca` attribute is an experimental feature [E0658]
    const BAR: bool;
}

impl Bar for bool {
    const BAR: bool = std::gca!(false);
    //~^ ERROR use of unstable library feature `min_generic_const_args` [E0658]
    //~| ERROR expected expression, found `gca!()` constant
    //~| ERROR implementation of a `#[rustc_always_gca]` must have a `gca!` RHS
}

fn main() {}
