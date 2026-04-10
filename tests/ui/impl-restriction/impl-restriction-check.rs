//@ aux-build: external-impl-restriction.rs
//@ revisions: e2015 e2018
//@ [e2015] edition: 2015
//@ [e2018] edition: 2018..
#![feature(impl_restriction)]
#![expect(incomplete_features)]

extern crate external_impl_restriction as external;

struct LocalType; // needed to avoid orphan rule errors

impl external::TopLevel for LocalType {} //~ ERROR trait cannot be implemented outside `external`
impl external::inner::Inner for LocalType {} //~ ERROR trait cannot be implemented outside `external`

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

    impl bar::Foo for i8 {} //[e2015]~ ERROR trait cannot be implemented outside `foo::bar`
    //[e2018]~^ ERROR trait cannot be implemented outside `crate::foo::bar`
    impl bar::Bar for i8 {} // OK
    impl bar::Baz for i8 {} // OK
    impl bar::Qux for i8 {} //[e2015]~ ERROR trait cannot be implemented outside `foo::bar`
    //[e2018]~^ ERROR trait cannot be implemented outside `crate::foo::bar`
    impl bar::FooBar for i8 {} // OK
}

impl foo::bar::Foo for u8 {} //[e2015]~ ERROR trait cannot be implemented outside `foo::bar`
//[e2018]~^ ERROR trait cannot be implemented outside `crate::foo::bar`
impl foo::bar::Bar for u8 {} //[e2015]~ ERROR trait cannot be implemented outside `foo`
//[e2018]~^ ERROR trait cannot be implemented outside `crate::foo`
impl foo::bar::Baz for u8 {} // OK
impl foo::bar::Qux for u8 {} //[e2015]~ ERROR trait cannot be implemented outside `foo::bar`
//[e2018]~^ ERROR trait cannot be implemented outside `crate::foo::bar`
impl foo::bar::FooBar for u8 {} //[e2015]~ ERROR trait cannot be implemented outside `foo`
//[e2018]~^ ERROR trait cannot be implemented outside `crate::foo`

fn main() {}
