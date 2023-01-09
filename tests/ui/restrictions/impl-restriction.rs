// compile-flags: --crate-type=lib
// aux-build: external-impl-restriction.rs

#![feature(impl_restriction)]

extern crate external_impl_restriction as external;

struct LocalType; // needed to avoid orphan rule errors

impl external::TopLevel for LocalType {} //~ ERROR trait cannot be implemented outside `external_impl_restriction`
impl external::inner::Inner for LocalType {} //~ ERROR trait cannot be implemented outside `external_impl_restriction`

pub mod foo {
    pub mod bar {
        pub(crate) impl(super) trait Foo {}
    }

    impl bar::Foo for i8 {}
}

impl foo::bar::Foo for u8 {} //~ ERROR trait cannot be implemented outside `foo`
