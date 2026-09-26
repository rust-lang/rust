#![feature(default_field_values)]

pub struct Bat {
    pub bax: u8 = panic!("asdf"),
    //~^ ERROR evaluation panicked: asdf
    pub bat: u8 = 130 + 130,
    //~^ ERROR attempt to compute `130_u8 + 130_u8`, which would overflow
}

pub struct Baz<const C: u8> {
    pub bax: u8 = 130 + C, //~ WARN
    pub bat: u8 = 130 + 130, //~ WARN
    // ^ If we run `const_eval_poly` without restricting const params, this would be
    // attempt to compute `130_u8 + 130_u8`, which would overflow
    pub bay: u8 = 1, //~ WARN
    pub bap: u8 = C, //~ WARN
    pub ban: u8 = panic!("asdf"),
    // ^ If we run `const_eval_poly` without restricting const params, this would be
    // evaluation panicked: asdf
    // FIXME: This whould WARN!
}

fn main() {}
