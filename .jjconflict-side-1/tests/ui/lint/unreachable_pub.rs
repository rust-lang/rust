//@ check-pass
//@ edition: 2018
//@ run-rustfix

#![allow(unused)]
#![warn(unreachable_pub)]

mod private_mod {
    // non-leaked `pub` items in private module should be linted
    pub use std::fmt; //~ WARNING unreachable_pub
    pub use std::env::{Args}; // braced-use has different item spans than unbraced
    //~^ WARNING unreachable_pub

    // we lint on struct definition
    pub struct Hydrogen { //~ WARNING unreachable_pub
        // but not on fields, even if they are `pub` as putting `pub(crate)`
        // it would clutter the source code for little value
        pub neutrons: usize,
        pub(crate) electrons: usize
    }
    pub(crate) struct Calcium {
        pub neutrons: usize,
    }
    impl Hydrogen {
        // impls, too
        pub fn count_neutrons(&self) -> usize { self.neutrons } //~ WARNING unreachable_pub
        pub(crate) fn count_electrons(&self) -> usize { self.electrons }
    }
    impl Clone for Hydrogen {
        fn clone(&self) -> Hydrogen {
            Hydrogen { neutrons: self.neutrons, electrons: self.electrons }
        }
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

    mod private_in_private {
        pub enum Helium {} //~ WARNING unreachable_pub
        pub fn beryllium() {} //~ WARNING unreachable_pub
    }

    pub(crate) mod crate_in_private {
        pub const CARBON: usize = 1; //~ WARNING unreachable_pub
    }

    pub mod pub_in_private { //~ WARNING unreachable_pub
        pub static NITROGEN: usize = 2; //~ WARNING unreachable_pub
    }

    fn foo() {
        const {
            pub struct Foo; //~ WARNING unreachable_pub
        };
    }

    enum Weird {
        Variant = {
            pub struct Foo; //~ WARNING unreachable_pub

            mod tmp {
                pub struct Bar; //~ WARNING unreachable_pub
            }

            let _ = tmp::Bar;

            0
        },
    }

    pub use fpu_precision::set_precision; //~ WARNING unreachable_pub

    mod fpu_precision {
        pub fn set_precision<T>() {} //~ WARNING unreachable_pub
        pub fn set_micro_precision<T>() {} //~ WARNING unreachable_pub
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
    let _ = private_mod::beryllium();
    let _ = private_mod::crate_in_private::CARBON;
    let _ = private_mod::pub_in_private::NITROGEN;
    let _ = unsafe { private_mod::catalyze() };
}
