// This is just like unreachable_pub.rs, but without the
// `crate_visibility_modifier` feature (so that we can test the suggestions to
// use `pub(crate)` that are given when that feature is off, as opposed to the
// suggestions to use `crate` given when it is on). When that feature becomes
// stable, this test can be deleted.

// check-pass


#![warn(unreachable_pub)]

mod private_mod {
    // non-leaked `pub` items in private module should be linted
    pub use std::fmt; //~ WARNING unreachable_pub
    pub use std::env::{Args}; // braced-use has different item spans than unbraced
    //~^ WARNING unreachable_pub

    pub struct Hydrogen { //~ WARNING unreachable_pub
        // `pub` struct fields, too
        pub neutrons: usize, //~ WARNING unreachable_pub
        // (... but not more-restricted fields)
        pub(crate) electrons: usize
    }
    impl Hydrogen {
        // impls, too
        pub fn count_neutrons(&self) -> usize { self.neutrons } //~ WARNING unreachable_pub
        pub(crate) fn count_electrons(&self) -> usize { self.electrons }
    }

    pub enum Helium {} //~ WARNING unreachable_pub
    pub union Lithium { c1: usize, c2: u8 } //~ WARNING unreachable_pub
    pub fn beryllium() {} //~ WARNING unreachable_pub
    pub trait Boron {} //~ WARNING unreachable_pub
    pub const CARBON: usize = 1; //~ WARNING unreachable_pub
    pub static NITROGEN: usize = 2; //~ WARNING unreachable_pub
    pub type Oxygen = bool; //~ WARNING unreachable_pub

    macro_rules! define_empty_struct_with_visibility {
        ($visibility: vis, $name: ident) => { $visibility struct $name {} }
        //~^ WARNING unreachable_pub
    }
    define_empty_struct_with_visibility!(pub, Fluorine);

    extern "C" {
        pub fn catalyze() -> bool; //~ WARNING unreachable_pub
    }

    // items leaked through signatures (see `get_neon` below) are OK
    pub struct Neon {}

    // crate-visible items are OK
    pub(crate) struct Sodium {}
}

pub mod public_mod {
    // module is public: these are OK, too
    pub struct Magnesium {}
    pub(crate) struct Aluminum {}
}

pub fn get_neon() -> private_mod::Neon {
    private_mod::Neon {}
}

fn main() {
    let _ = get_neon();
}
