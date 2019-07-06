// This is just like unreachable_pub.rs, but without the
// `crate_visibility_modifier` feature (so that we can test the suggestions to
// use `pub(crate)` that are given when that feature is off, as opposed to the
// suggestions to use `crate` given when it is on). When that feature becomes
// stable, this test can be deleted.

// build-pass (FIXME(62277): could be check-pass?)


#![allow(unused)]
#![warn(unreachable_pub)]

mod private_mod {
    // non-leaked `pub` items in private module should be linted
    pub use std::fmt;
    pub use std::env::{Args}; // braced-use has different item spans than unbraced

    pub struct Hydrogen {
        // `pub` struct fields, too
        pub neutrons: usize,
        // (... but not more-restricted fields)
        pub(crate) electrons: usize
    }
    impl Hydrogen {
        // impls, too
        pub fn count_neutrons(&self) -> usize { self.neutrons }
        pub(crate) fn count_electrons(&self) -> usize { self.electrons }
    }

    pub enum Helium {}
    pub union Lithium { c1: usize, c2: u8 }
    pub fn beryllium() {}
    pub trait Boron {}
    pub const CARBON: usize = 1;
    pub static NITROGEN: usize = 2;
    pub type Oxygen = bool;

    macro_rules! define_empty_struct_with_visibility {
        ($visibility: vis, $name: ident) => { $visibility struct $name {} }
    }
    define_empty_struct_with_visibility!(pub, Fluorine);

    extern {
        pub fn catalyze() -> bool;
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
