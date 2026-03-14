//@ compile-flags: --crate-type=lib

// Visibily cannot resolve `use`d paths, whereas restrictions can.
// <https://github.com/rust-lang/rust/issues/60552>

#![feature(impl_restriction)]
#![expect(incomplete_features)]

mod m1 {
    pub impl(in crate::m2) trait T1 {} // OK
}

mod m3 {
    mod m4 {
        pub impl(in crate::m2) trait T2 {} //~ ERROR trait implementation can only be restricted to ancestor modules
        pub impl(in m6) trait T3 {} // OK
        pub impl(in m6::m5) trait T4 {} //~ ERROR trait implementation can only be restricted to ancestor modules
        pub impl(in m7) trait T5 {} //~ ERROR expected module, found enum `::m7` [E0577]
    }
    mod m5 {}
    pub enum E {}
}

use m1 as m2;
use crate::m3 as m6;
use crate::m3::E as m7;
