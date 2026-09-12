//@ run-rustfix
//@ edition: 2021

// Nested macro import suggestions should preserve the original `use` visibility.

#![allow(unused)]

mod a {
    #[macro_export]
    macro_rules! m_pub {
        () => {};
    }
    #[macro_export]
    macro_rules! m_crate {
        () => {};
    }
    #[macro_export]
    macro_rules! m_in {
        () => {};
    }
    #[macro_export]
    macro_rules! m_super {
        () => {};
    }
    #[macro_export]
    macro_rules! m_self {
        () => {};
    }
    #[macro_export]
    macro_rules! m_private {
        () => {};
    }

    pub struct S;

    mod b0 {
        pub use super::{m_pub, S as PubS};
        //~^ ERROR unresolved import `super::m_pub`
        //~| HELP a macro with this name exists at the root of the crate
    }

    mod b1 {
        pub(crate) use super::{m_crate, S as CrateS};
        //~^ ERROR unresolved import `super::m_crate`
        //~| HELP a macro with this name exists at the root of the crate
    }

    mod b2 {
        pub(in crate::a) use super::{m_in, S as InS};
        //~^ ERROR unresolved import `super::m_in`
        //~| HELP a macro with this name exists at the root of the crate
    }

    mod b3 {
        pub(super) use super::{m_super, S as SuperS};
        //~^ ERROR unresolved import `super::m_super`
        //~| HELP a macro with this name exists at the root of the crate
    }

    mod b4 {
        pub(self) use super::{m_self, S as SelfS};
        //~^ ERROR unresolved import `super::m_self`
        //~| HELP a macro with this name exists at the root of the crate
    }

    mod b5 {
        use super::{m_private, S as PrivateS};
        //~^ ERROR unresolved import `super::m_private`
        //~| HELP a macro with this name exists at the root of the crate
    }
}

fn main() {}
