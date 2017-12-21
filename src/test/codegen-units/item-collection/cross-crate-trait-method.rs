// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-tidy-linelength
// compile-flags:-Zprint-trans-items=eager

#![deny(dead_code)]

// aux-build:cgu_export_trait_method.rs
extern crate cgu_export_trait_method;

use cgu_export_trait_method::Trait;

//~ TRANS_ITEM fn alloc::allocator[0]::{{impl}}[0]::align[0] @@ cross_crate_trait_method0[Internal]
//~ TRANS_ITEM fn alloc::allocator[0]::{{impl}}[0]::from_size_align_unchecked[0] @@ cross_crate_trait_method0[Internal]
//~ TRANS_ITEM fn alloc::allocator[0]::{{impl}}[0]::size[0] @@ cross_crate_trait_method0[Internal]
//~ TRANS_ITEM fn alloc::heap[0]::box_free[0]<core::any[0]::Any[0]> @@ cross_crate_trait_method0[Internal]
//~ TRANS_ITEM fn alloc::heap[0]::{{impl}}[0]::dealloc[0] @@ cross_crate_trait_method0[Internal]
//~ TRANS_ITEM fn core::mem[0]::uninitialized[0]<std::rt[0]::lang_start[0]::{{closure}}[0]<()>> @@ cross_crate_trait_method0[Internal]
//~ TRANS_ITEM fn core::ptr[0]::drop_in_place[0]<alloc::boxed[0]::Box[0]<core::any[0]::Any[0]>> @@ cross_crate_trait_method0[Internal]
//~ TRANS_ITEM fn core::ptr[0]::drop_in_place[0]<core::any[0]::Any[0]> @@ cross_crate_trait_method0[Internal]
//~ TRANS_ITEM fn core::ptr[0]::drop_in_place[0]<core::result[0]::Result[0]<i32, alloc::boxed[0]::Box[0]<core::any[0]::Any[0]>>> @@ cross_crate_trait_method0[Internal]
//~ TRANS_ITEM fn core::ptr[0]::read[0]<std::rt[0]::lang_start[0]::{{closure}}[0]<()>> @@ cross_crate_trait_method0[Internal]
//~ TRANS_ITEM fn core::ptr[0]::write[0]<i32> @@ cross_crate_trait_method0[Internal]
//~ TRANS_ITEM fn core::result[0]::{{impl}}[0]::unwrap_or[0]<i32, alloc::boxed[0]::Box[0]<core::any[0]::Any[0]>> @@ cross_crate_trait_method0[Internal]
//~ TRANS_ITEM fn std::panic[0]::catch_unwind[0]<std::rt[0]::lang_start[0]::{{closure}}[0]<()>, i32> @@ cross_crate_trait_method0[Internal]
//~ TRANS_ITEM fn std::panicking[0]::try[0]::do_call[0]<std::rt[0]::lang_start[0]::{{closure}}[0]<()>, i32> @@ cross_crate_trait_method0[Internal]
//~ TRANS_ITEM fn std::panicking[0]::try[0]<i32, std::rt[0]::lang_start[0]::{{closure}}[0]<()>> @@ cross_crate_trait_method0[Internal]
//~ TRANS_ITEM fn std::rt[0]::lang_start[0]::{{closure}}[0]::{{closure}}[0]<(), i32, extern "rust-call" fn(()) -> i32, fn()> @@ cross_crate_trait_method0[Internal]
//~ TRANS_ITEM fn std::rt[0]::lang_start[0]::{{closure}}[0]<(), i32, extern "rust-call" fn(()) -> i32, &fn()> @@ cross_crate_trait_method0[Internal]
//~ TRANS_ITEM fn std::rt[0]::lang_start[0]<()> @@ cross_crate_trait_method0[External]
//~ TRANS_ITEM fn std::sys_common[0]::backtrace[0]::__rust_begin_short_backtrace[0]<std::rt[0]::lang_start[0]::{{closure}}[0]::{{closure}}[0]<()>, i32> @@ cross_crate_trait_method0[Internal]
//~ TRANS_ITEM fn cross_crate_trait_method::main[0]
fn main()
{
    // The object code of these methods is contained in the external crate, so
    // calling them should *not* introduce codegen items in the current crate.
    let _: (u32, u32) = Trait::without_default_impl(0);
    let _: (char, u32) = Trait::without_default_impl(0);

    // Currently, no object code is generated for trait methods with default
    // implemenations, unless they are actually called from somewhere. Therefore
    // we cannot import the implementations and have to create our own inline.
    //~ TRANS_ITEM fn cgu_export_trait_method::Trait[0]::with_default_impl[0]<u32>
    let _ = Trait::with_default_impl(0u32);
    //~ TRANS_ITEM fn cgu_export_trait_method::Trait[0]::with_default_impl[0]<char>
    let _ = Trait::with_default_impl('c');



    //~ TRANS_ITEM fn cgu_export_trait_method::Trait[0]::with_default_impl_generic[0]<u32, &str>
    let _ = Trait::with_default_impl_generic(0u32, "abc");
    //~ TRANS_ITEM fn cgu_export_trait_method::Trait[0]::with_default_impl_generic[0]<u32, bool>
    let _ = Trait::with_default_impl_generic(0u32, false);

    //~ TRANS_ITEM fn cgu_export_trait_method::Trait[0]::with_default_impl_generic[0]<char, i16>
    let _ = Trait::with_default_impl_generic('x', 1i16);
    //~ TRANS_ITEM fn cgu_export_trait_method::Trait[0]::with_default_impl_generic[0]<char, i32>
    let _ = Trait::with_default_impl_generic('y', 0i32);

    //~ TRANS_ITEM fn cgu_export_trait_method::{{impl}}[1]::without_default_impl_generic[0]<char>
    let _: (u32, char) = Trait::without_default_impl_generic('c');
    //~ TRANS_ITEM fn cgu_export_trait_method::{{impl}}[1]::without_default_impl_generic[0]<bool>
    let _: (u32, bool) = Trait::without_default_impl_generic(false);

    //~ TRANS_ITEM fn cgu_export_trait_method::{{impl}}[0]::without_default_impl_generic[0]<char>
    let _: (char, char) = Trait::without_default_impl_generic('c');
    //~ TRANS_ITEM fn cgu_export_trait_method::{{impl}}[0]::without_default_impl_generic[0]<bool>
    let _: (char, bool) = Trait::without_default_impl_generic(false);
}
