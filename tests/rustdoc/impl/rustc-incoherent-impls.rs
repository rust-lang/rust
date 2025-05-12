//@ aux-build:incoherent-impl-types.rs
//@ build-aux-docs

#![crate_name = "foo"]
#![feature(rustc_attrs)]

extern crate incoherent_impl_types;

// The only way this actually shows up is if the type gets inlined.
#[doc(inline)]
pub use incoherent_impl_types::FooTrait;

//@ has foo/trait.FooTrait.html
//@ count - '//section[@id="method.do_something"]' 1
impl dyn FooTrait {
    #[rustc_allow_incoherent_impl]
    pub fn do_something() {}
}

#[doc(inline)]
pub use incoherent_impl_types::FooStruct;

//@ has foo/struct.FooStruct.html
//@ count - '//section[@id="method.do_something"]' 1
impl FooStruct {
    #[rustc_allow_incoherent_impl]
    pub fn do_something() {}
}
