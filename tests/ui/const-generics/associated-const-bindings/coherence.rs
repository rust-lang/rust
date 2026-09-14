#![feature(min_generic_const_args)]
#![expect(incomplete_features)]

pub trait IsVoid {
    #[rustc_always_gca]
    const IS_VOID: bool;
}
impl IsVoid for () {
    const IS_VOID: bool = core::direct_const_arg!(true);
}

pub trait Maybe {}
impl Maybe for () {}
impl Maybe for () where (): IsVoid<IS_VOID = true> {}
//~^ ERROR conflicting implementations of trait `Maybe` for type `()`

fn main() {}
