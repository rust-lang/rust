//@ compile-flags:-Clink-dead-code -Zinline-mir=no -Copt-level=0

#![deny(dead_code)]
#![crate_type = "lib"]

//@ aux-build:cgu_export_trait_method.rs
extern crate cgu_export_trait_method;

use cgu_export_trait_method::Trait;

//~ MONO_ITEM fn start
#[no_mangle]
pub fn start(_: isize, _: *const *const u8) -> isize {
    // The object code of these methods is contained in the external crate, so
    // calling them should *not* introduce codegen items in the current crate.
    let _: (u32, u32) = Trait::without_default_impl(0);
    let _: (char, u32) = Trait::without_default_impl(0);

    // Currently, no object code is generated for trait methods with default
    // implementations, unless they are actually called from somewhere. Therefore
    // we cannot import the implementations and have to create our own inline.
    //~ MONO_ITEM fn <u32 as cgu_export_trait_method::Trait>::with_default_impl
    let _ = Trait::with_default_impl(0u32);
    //~ MONO_ITEM fn <char as cgu_export_trait_method::Trait>::with_default_impl
    let _ = Trait::with_default_impl('c');

    //~ MONO_ITEM fn <u32 as cgu_export_trait_method::Trait>::with_default_impl_generic::<&str>
    let _ = Trait::with_default_impl_generic(0u32, "abc");
    //~ MONO_ITEM fn <u32 as cgu_export_trait_method::Trait>::with_default_impl_generic::<bool>
    let _ = Trait::with_default_impl_generic(0u32, false);

    //~ MONO_ITEM fn <char as cgu_export_trait_method::Trait>::with_default_impl_generic::<i16>
    let _ = Trait::with_default_impl_generic('x', 1i16);
    //~ MONO_ITEM fn <char as cgu_export_trait_method::Trait>::with_default_impl_generic::<i32>
    let _ = Trait::with_default_impl_generic('y', 0i32);

    //~ MONO_ITEM fn <u32 as cgu_export_trait_method::Trait>::without_default_impl_generic::<char>
    let _: (u32, char) = Trait::without_default_impl_generic('c');
    //~ MONO_ITEM fn <u32 as cgu_export_trait_method::Trait>::without_default_impl_generic::<bool>
    let _: (u32, bool) = Trait::without_default_impl_generic(false);

    //~ MONO_ITEM fn <char as cgu_export_trait_method::Trait>::without_default_impl_generic::<char>
    let _: (char, char) = Trait::without_default_impl_generic('c');
    //~ MONO_ITEM fn <char as cgu_export_trait_method::Trait>::without_default_impl_generic::<bool>
    let _: (char, bool) = Trait::without_default_impl_generic(false);

    0
}
