//@ aux-build: external-impl-restriction.rs
#![feature(impl_restriction)]
#![expect(incomplete_features)]

extern crate external_impl_restriction as external;

struct LocalType; // needed to avoid orphan rule errors

impl external::TopLevel for LocalType {} //~ ERROR trait cannot be implemented outside `external_impl_restriction`
impl external::inner::Inner for LocalType {} //~ ERROR trait cannot be implemented outside `external_impl_restriction`

pub mod foo {
    pub mod bar {
        pub(crate) impl(self) trait Foo {}
        pub(crate) impl(super) trait Bar {}
        pub impl(crate) trait Baz {}
        pub(crate) impl(in crate::foo::bar) trait Qux {}
        pub(crate) impl(in crate::foo) trait FooBar {}

        impl Foo for i16 {} // OK
        impl Bar for i16 {} // OK
        impl Baz for i16 {} // OK
        impl Qux for i16 {} // OK
        impl FooBar for i16 {} // OK
    }

    impl bar::Foo for i8 {} //~ ERROR trait cannot be implemented outside `bar`
    impl bar::Bar for i8 {} // OK
    impl bar::Baz for i8 {} // OK
    impl bar::Qux for i8 {} //~ ERROR trait cannot be implemented outside `bar`
    impl bar::FooBar for i8 {} // OK
}

impl foo::bar::Foo for u8 {} //~ ERROR trait cannot be implemented outside `bar`
impl foo::bar::Bar for u8 {} //~ ERROR trait cannot be implemented outside `foo`
impl foo::bar::Baz for u8 {} // OK
impl foo::bar::Qux for u8 {} //~ ERROR trait cannot be implemented outside `bar`
impl foo::bar::FooBar for u8 {} //~ ERROR trait cannot be implemented outside `foo`

fn main() {}
