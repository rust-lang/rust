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

// aux-build:cgu_generic_function.rs
extern crate cgu_generic_function;

//~ TRANS_ITEM fn alloc::allocator[0]::{{impl}}[0]::align[0] @@ cross_crate_generic_functions0[Internal]
//~ TRANS_ITEM fn alloc::allocator[0]::{{impl}}[0]::from_size_align_unchecked[0] @@ cross_crate_generic_functions0[Internal]
//~ TRANS_ITEM fn alloc::allocator[0]::{{impl}}[0]::size[0] @@ cross_crate_generic_functions0[Internal]
//~ TRANS_ITEM fn alloc::heap[0]::box_free[0]<core::any[0]::Any[0]> @@ cross_crate_generic_functions0[Internal]
//~ TRANS_ITEM fn alloc::heap[0]::{{impl}}[0]::dealloc[0] @@ cross_crate_generic_functions0[Internal]
//~ TRANS_ITEM fn core::mem[0]::uninitialized[0]<std::rt[0]::lang_start[0]::{{closure}}[0]<()>> @@ cross_crate_generic_functions0[Internal]
//~ TRANS_ITEM fn core::ptr[0]::drop_in_place[0]<alloc::boxed[0]::Box[0]<core::any[0]::Any[0]>> @@ cross_crate_generic_functions0[Internal]
//~ TRANS_ITEM fn core::ptr[0]::drop_in_place[0]<core::any[0]::Any[0]> @@ cross_crate_generic_functions0[Internal]
//~ TRANS_ITEM fn core::ptr[0]::drop_in_place[0]<core::result[0]::Result[0]<i32, alloc::boxed[0]::Box[0]<core::any[0]::Any[0]>>> @@ cross_crate_generic_functions0[Internal]
//~ TRANS_ITEM fn core::ptr[0]::read[0]<std::rt[0]::lang_start[0]::{{closure}}[0]<()>> @@ cross_crate_generic_functions0[Internal]
//~ TRANS_ITEM fn core::ptr[0]::write[0]<i32> @@ cross_crate_generic_functions0[Internal]
//~ TRANS_ITEM fn core::result[0]::{{impl}}[0]::unwrap_or[0]<i32, alloc::boxed[0]::Box[0]<core::any[0]::Any[0]>> @@ cross_crate_generic_functions0[Internal]
//~ TRANS_ITEM fn std::panic[0]::catch_unwind[0]<std::rt[0]::lang_start[0]::{{closure}}[0]<()>, i32> @@ cross_crate_generic_functions0[Internal]
//~ TRANS_ITEM fn std::panicking[0]::try[0]::do_call[0]<std::rt[0]::lang_start[0]::{{closure}}[0]<()>, i32> @@ cross_crate_generic_functions0[Internal]
//~ TRANS_ITEM fn std::panicking[0]::try[0]<i32, std::rt[0]::lang_start[0]::{{closure}}[0]<()>> @@ cross_crate_generic_functions0[Internal]
//~ TRANS_ITEM fn std::rt[0]::lang_start[0]::{{closure}}[0]::{{closure}}[0]<(), i32, extern "rust-call" fn(()) -> i32, fn()> @@ cross_crate_generic_functions0[Internal]
//~ TRANS_ITEM fn std::rt[0]::lang_start[0]::{{closure}}[0]<(), i32, extern "rust-call" fn(()) -> i32, &fn()> @@ cross_crate_generic_functions0[Internal]
//~ TRANS_ITEM fn std::rt[0]::lang_start[0]<()> @@ cross_crate_generic_functions0[External]
//~ TRANS_ITEM fn std::sys_common[0]::backtrace[0]::__rust_begin_short_backtrace[0]<std::rt[0]::lang_start[0]::{{closure}}[0]::{{closure}}[0]<()>, i32> @@ cross_crate_generic_functions0[Internal]
//~ TRANS_ITEM fn cross_crate_generic_functions::main[0]
fn main()
{
    //~ TRANS_ITEM fn cgu_generic_function::bar[0]<u32>
    //~ TRANS_ITEM fn cgu_generic_function::foo[0]<u32>
    let _ = cgu_generic_function::foo(1u32);

    //~ TRANS_ITEM fn cgu_generic_function::bar[0]<u64>
    //~ TRANS_ITEM fn cgu_generic_function::foo[0]<u64>
    let _ = cgu_generic_function::foo(2u64);

    // This should not introduce a codegen item
    let _ = cgu_generic_function::exported_but_not_generic(3);
}
