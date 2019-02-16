// Extern crate items are marked as used if they are used
// through extern prelude entries introduced by them.

// edition:2018

#![deny(unused_extern_crates)]

extern crate core as iso1; //~ ERROR `extern crate` is not idiomatic in the new edition
extern crate core as iso2; //~ ERROR `extern crate` is not idiomatic in the new edition
extern crate core as iso3; //~ ERROR `extern crate` is not idiomatic in the new edition
extern crate core as iso4; //~ ERROR `extern crate` is not idiomatic in the new edition

// Doesn't introduce its extern prelude entry, so it's still considered unused.
extern crate core; //~ ERROR unused extern crate

mod m {
    use iso1::any as are_you_okay1;
    use ::iso2::any as are_you_okay2;
    type AreYouOkay1 = iso3::any::Any;
    type AreYouOkay2 = ::iso4::any::Any;

    use core::any as are_you_okay3;
    use ::core::any as are_you_okay4;
    type AreYouOkay3 = core::any::Any;
    type AreYouOkay4 = ::core::any::Any;
}

fn main() {}
