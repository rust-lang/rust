//@ compile-flags: --crate-type=lib
//@ edition: 2018

#![feature(impl_restriction)]
#![expect(incomplete_features)]

mod a {
    mod b {
        pub impl(in a::b) trait T1 {} //~ ERROR relative paths are not supported in `impl` restrictions in 2018 edition or later

        pub impl(in ::core) trait T2 {} //~ ERROR trait implementation can only be restricted to ancestor modules

        // OK paths
        pub impl(crate) trait T3 {}
        pub impl(self) trait T4 {}
        pub impl(super) trait T5 {}
        pub impl(in crate::a) trait T6 {}
        pub impl(in super::super) trait T7 {}
    }
}

pub impl(crate) trait T8 {}
pub impl(self) trait T9 {}

pub impl(super) trait T10 {} //~ ERROR too many leading `super` keywords [E0433]
