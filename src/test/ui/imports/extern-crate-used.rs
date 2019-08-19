// Extern crate items are marked as used if they are used
// through extern prelude entries introduced by them.

// edition:2018

#![deny(unused_extern_crates)]

// Shouldn't suggest changing to `use`, as new name
// would no longer be added to the prelude which could cause
// compilation errors for imports that use the new name in
// other modules. See #57672.
extern crate core as iso1;
extern crate core as iso2;
extern crate core as iso3;
extern crate core as iso4;

// Doesn't introduce its extern prelude entry, so it's still considered unused.
extern crate core; //~ ERROR unused extern crate

mod m {
    use iso1::any as are_you_okay1;
    use ::iso2::any as are_you_okay2;
    type AreYouOkay1 = dyn iso3::any::Any;
    type AreYouOkay2 = dyn (::iso4::any::Any);

    use core::any as are_you_okay3;
    use ::core::any as are_you_okay4;
    type AreYouOkay3 = dyn core::any::Any;
    type AreYouOkay4 = dyn (::core::any::Any);
}

fn main() {}
