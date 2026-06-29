//@ aux-build: external-impl-restriction.rs
#![feature(impl_restriction)]

extern crate external_impl_restriction as external;

pub mod a {
    pub enum E {}
    pub mod d {}
    pub mod b {
        pub mod c {}

        // We do not use crate-relative paths here, since we follow the
        // "uniform paths" approach used for type/expression paths.
        pub impl(in a::b) trait T1 {} //~ ERROR cannot find module or crate `a` in this scope [E0433]

        pub impl(in ::std) trait T2 {} //~ ERROR trait implementation can only be restricted to ancestor modules

        pub impl(in self::c) trait T3 {} //~ ERROR trait implementation can only be restricted to ancestor modules

        pub impl(in super::d) trait T4 {} //~ ERROR trait implementation can only be restricted to ancestor modules

        pub impl(in crate::c) trait T5 {} //~ ERROR cannot find module `c` in the crate root [E0433]

        pub impl(in super::E) trait T6 {} //~ ERROR expected module, found enum `super::E` [E0577]

        pub impl(in super::super::super) trait T7 {} //~ ERROR too many leading `super` keywords [E0433]

        // OK paths
        pub impl(crate) trait T8 {}
        pub impl(self) trait T9 {}
        pub impl(super) trait T10 {}
        pub impl(in crate::a) trait T11 {}
        pub impl(in super::super) trait T12 {}

        // Check if we can resolve paths referring to modules declared later.
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

pub impl(in external) trait T18 {} //~ ERROR trait implementation can only be restricted to ancestor modules

// Check if we can resolve paths referring to modules declared later.
pub impl(in crate::j) trait L4 {} //~ ERROR trait implementation can only be restricted to ancestor modules

pub impl(in crate::I) trait L5 {} //~ ERROR expected module, found enum `crate::I` [E0577]

pub enum I {}
pub mod j {}

// Check if we can resolve `use`d paths.
mod m1 {
    pub impl(in crate::m2) trait U1 {} // OK
}

use m1 as m2;

mod m3 {
    mod m4 {
        pub impl(in crate::m2) trait U2 {} //~ ERROR trait implementation can only be restricted to ancestor modules
        pub impl(in m6) trait U3 {} // OK
        pub impl(in m6::m5) trait U4 {} //~ ERROR trait implementation can only be restricted to ancestor modules
        pub impl(in m7) trait U5 {} //~ ERROR expected module, found enum `m7` [E0577]

        use crate::m3 as m6;
        use crate::m3::E as m7;
    }
    mod m5 {}
    pub enum E {}
}

fn main() {}
