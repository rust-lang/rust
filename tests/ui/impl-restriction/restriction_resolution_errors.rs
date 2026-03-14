//@ compile-flags: --crate-type=lib
//@ edition: 2015

#![feature(impl_restriction)]
#![expect(incomplete_features)]

pub mod a {
    pub enum E {}
    pub mod d {}
    pub mod b {
        pub mod c {}
        pub impl(in a::b) trait T1 {} // OK for 2015 edition

        pub impl(in ::core) trait T2 {} //~ ERROR cannot find `core` in the crate root [E0433]

        pub impl(in self::c) trait T3 {} //~ ERROR trait implementation can only be restricted to ancestor modules

        pub impl(in super::d) trait T4 {} //~ ERROR trait implementation can only be restricted to ancestor modules

        pub impl(in crate::c) trait T5 {} //~ ERROR cannot find module or crate `c` in `crate` [E0433]

        pub impl(in super::E) trait T6 {} //~ ERROR expected module, found enum `super::E` [E0577]

        pub impl(in super::super::super) trait T7 {} //~ ERROR too many leading `super` keywords [E0433]

        // OK paths
        pub impl(crate) trait T8 {}
        pub impl(self) trait T9 {}
        pub impl(super) trait T10 {}
        pub impl(in crate::a) trait T11 {}
        pub impl(in super::super) trait T12 {}

        // Visibility cannot resolve paths declared later, whereas restrictions can.
        pub impl(in self::f) trait L1 {} //~ ERROR trait implementation can only be restricted to ancestor modules

        pub impl(in super::G) trait L2 {} //~ ERROR expected module, found enum `super::G` [E0577]

        pub impl(in super::h) trait L3 {} //~ ERROR trait implementation can only be restricted to ancestor modules

        pub mod f {}
    }

    pub enum G {}
    pub mod h {}
}

pub impl(in crate::a) trait T13 {} //~ ERROR trait implementation can only be restricted to ancestor modules

pub impl(in crate::a::E) trait T14 {} //~ ERROR expected module, found enum `crate::a::E` [E0577]

pub impl(crate) trait T15 {}
pub impl(self) trait T16 {}

pub impl(super) trait T17 {} //~ ERROR too many leading `super` keywords [E0433]

// Visibility cannot resolve paths declared later, whereas restrictions can.
pub impl(in crate::j) trait L4 {} //~ ERROR trait implementation can only be restricted to ancestor modules

pub impl(in crate::I) trait L5 {} //~ ERROR expected module, found enum `crate::I` [E0577]

pub enum I {}
pub mod j {}
