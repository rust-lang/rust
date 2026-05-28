//@ check-pass
// (this requires debug assertions)

#![feature(adt_const_params, unsized_const_params)]
#![allow(incomplete_features)]

pub const BAR: () = ice::<"">();
pub const fn ice<const N: &'static str>() {
    &10;
}

fn main() {}
