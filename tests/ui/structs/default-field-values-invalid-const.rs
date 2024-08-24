#![feature(default_field_values, generic_const_exprs)]
#![allow(incomplete_features)]

#[warn(default_field_always_invalid_const)] //~ WARN lint `default_field_always_invalid_const` can't be warned on
pub struct Bat {
    pub bax: u8 = panic!("asdf"),
    //~^ ERROR evaluation of constant value failed
    //~| WARN default field fails const-evaluation
}

pub struct Baz<const C: u8> {
    pub bax: u8 = 130 + C, // ok
    pub bat: u8 = 130 + 130,
    //~^ ERROR evaluation of `Baz::<C>::bat::{constant#0}` failed
    //~| ERROR default field fails const-evaluation
    pub bay: u8 = 1, // ok
}

fn main() {}
