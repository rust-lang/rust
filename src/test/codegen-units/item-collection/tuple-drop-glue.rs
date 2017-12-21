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
// compile-flags:-Zinline-in-all-cgus

#![deny(dead_code)]

//~ TRANS_ITEM fn core::ptr[0]::drop_in_place[0]<tuple_drop_glue::Dropped[0]> @@ tuple_drop_glue0[Internal]
struct Dropped;

impl Drop for Dropped {
    //~ TRANS_ITEM fn tuple_drop_glue::{{impl}}[0]::drop[0]
    fn drop(&mut self) {}
}

//~ TRANS_ITEM fn alloc::allocator[0]::{{impl}}[0]::align[0] @@ tuple_drop_glue0[Internal]
//~ TRANS_ITEM fn alloc::allocator[0]::{{impl}}[0]::from_size_align_unchecked[0] @@ tuple_drop_glue0[Internal]
//~ TRANS_ITEM fn alloc::allocator[0]::{{impl}}[0]::size[0] @@ tuple_drop_glue0[Internal]
//~ TRANS_ITEM fn alloc::heap[0]::box_free[0]<core::any[0]::Any[0]> @@ tuple_drop_glue0[Internal]
//~ TRANS_ITEM fn alloc::heap[0]::{{impl}}[0]::dealloc[0] @@ tuple_drop_glue0[Internal]
//~ TRANS_ITEM fn core::mem[0]::uninitialized[0]<std::rt[0]::lang_start[0]::{{closure}}[0]<()>> @@ tuple_drop_glue0[Internal]
//~ TRANS_ITEM fn core::ptr[0]::drop_in_place[0]<alloc::boxed[0]::Box[0]<core::any[0]::Any[0]>> @@ tuple_drop_glue0[Internal]
//~ TRANS_ITEM fn core::ptr[0]::drop_in_place[0]<core::any[0]::Any[0]> @@ tuple_drop_glue0[Internal]
//~ TRANS_ITEM fn core::ptr[0]::drop_in_place[0]<core::result[0]::Result[0]<i32, alloc::boxed[0]::Box[0]<core::any[0]::Any[0]>>> @@ tuple_drop_glue0[Internal]
//~ TRANS_ITEM fn core::ptr[0]::read[0]<std::rt[0]::lang_start[0]::{{closure}}[0]<()>> @@ tuple_drop_glue0[Internal]
//~ TRANS_ITEM fn core::ptr[0]::write[0]<i32> @@ tuple_drop_glue0[Internal]
//~ TRANS_ITEM fn core::result[0]::{{impl}}[0]::unwrap_or[0]<i32, alloc::boxed[0]::Box[0]<core::any[0]::Any[0]>> @@ tuple_drop_glue0[Internal]
//~ TRANS_ITEM fn std::panic[0]::catch_unwind[0]<std::rt[0]::lang_start[0]::{{closure}}[0]<()>, i32> @@ tuple_drop_glue0[Internal]
//~ TRANS_ITEM fn std::panicking[0]::try[0]::do_call[0]<std::rt[0]::lang_start[0]::{{closure}}[0]<()>, i32> @@ tuple_drop_glue0[Internal]
//~ TRANS_ITEM fn std::panicking[0]::try[0]<i32, std::rt[0]::lang_start[0]::{{closure}}[0]<()>> @@ tuple_drop_glue0[Internal]
//~ TRANS_ITEM fn std::rt[0]::lang_start[0]::{{closure}}[0]::{{closure}}[0]<(), i32, extern "rust-call" fn(()) -> i32, fn()> @@ tuple_drop_glue0[Internal]
//~ TRANS_ITEM fn std::rt[0]::lang_start[0]::{{closure}}[0]<(), i32, extern "rust-call" fn(()) -> i32, &fn()> @@ tuple_drop_glue0[Internal]
//~ TRANS_ITEM fn std::rt[0]::lang_start[0]<()> @@ tuple_drop_glue0[External]
//~ TRANS_ITEM fn std::sys_common[0]::backtrace[0]::__rust_begin_short_backtrace[0]<std::rt[0]::lang_start[0]::{{closure}}[0]::{{closure}}[0]<()>, i32> @@ tuple_drop_glue0[Internal]
//~ TRANS_ITEM fn tuple_drop_glue::main[0]
fn main() {
    //~ TRANS_ITEM fn core::ptr[0]::drop_in_place[0]<(u32, tuple_drop_glue::Dropped[0])> @@ tuple_drop_glue0[Internal]
    let x = (0u32, Dropped);

    //~ TRANS_ITEM fn core::ptr[0]::drop_in_place[0]<(i16, (tuple_drop_glue::Dropped[0], bool))> @@ tuple_drop_glue0[Internal]
    //~ TRANS_ITEM fn core::ptr[0]::drop_in_place[0]<(tuple_drop_glue::Dropped[0], bool)> @@ tuple_drop_glue0[Internal]
    let x = (0i16, (Dropped, true));
}
