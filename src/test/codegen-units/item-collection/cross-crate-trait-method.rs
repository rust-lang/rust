// compile-flags:-Zprint-mono-items=eager

#![deny(dead_code)]
#![feature(start)]

// aux-build:cgu_export_trait_method.rs
extern crate cgu_export_trait_method;

use cgu_export_trait_method::Trait;

//~ MONO_ITEM fn cross_crate_trait_method::start[0]
#[start]
fn start(_: isize, _: *const *const u8) -> isize {
    // The object code of these methods is contained in the external crate, so
    // calling them should *not* introduce codegen items in the current crate.
    let _: (u32, u32) = Trait::without_default_impl(0);
    let _: (char, u32) = Trait::without_default_impl(0);

    // Currently, no object code is generated for trait methods with default
    // implementations, unless they are actually called from somewhere. Therefore
    // we cannot import the implementations and have to create our own inline.
    //~ MONO_ITEM fn cgu_export_trait_method::Trait[0]::with_default_impl[0]<u32>
    let _ = Trait::with_default_impl(0u32);
    //~ MONO_ITEM fn cgu_export_trait_method::Trait[0]::with_default_impl[0]<char>
    let _ = Trait::with_default_impl('c');



    //~ MONO_ITEM fn cgu_export_trait_method::Trait[0]::with_default_impl_generic[0]<u32, &str>
    let _ = Trait::with_default_impl_generic(0u32, "abc");
    //~ MONO_ITEM fn cgu_export_trait_method::Trait[0]::with_default_impl_generic[0]<u32, bool>
    let _ = Trait::with_default_impl_generic(0u32, false);

    //~ MONO_ITEM fn cgu_export_trait_method::Trait[0]::with_default_impl_generic[0]<char, i16>
    let _ = Trait::with_default_impl_generic('x', 1i16);
    //~ MONO_ITEM fn cgu_export_trait_method::Trait[0]::with_default_impl_generic[0]<char, i32>
    let _ = Trait::with_default_impl_generic('y', 0i32);

    //~ MONO_ITEM fn cgu_export_trait_method::{{impl}}[1]::without_default_impl_generic[0]<char>
    let _: (u32, char) = Trait::without_default_impl_generic('c');
    //~ MONO_ITEM fn cgu_export_trait_method::{{impl}}[1]::without_default_impl_generic[0]<bool>
    let _: (u32, bool) = Trait::without_default_impl_generic(false);

    //~ MONO_ITEM fn cgu_export_trait_method::{{impl}}[0]::without_default_impl_generic[0]<char>
    let _: (char, char) = Trait::without_default_impl_generic('c');
    //~ MONO_ITEM fn cgu_export_trait_method::{{impl}}[0]::without_default_impl_generic[0]<bool>
    let _: (char, bool) = Trait::without_default_impl_generic(false);

    0
}
