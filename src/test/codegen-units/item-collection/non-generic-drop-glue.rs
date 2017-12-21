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

//~ TRANS_ITEM fn core::ptr[0]::drop_in_place[0]<non_generic_drop_glue::StructWithDrop[0]> @@ non_generic_drop_glue0[Internal]
struct StructWithDrop {
    x: i32
}

impl Drop for StructWithDrop {
    //~ TRANS_ITEM fn non_generic_drop_glue::{{impl}}[0]::drop[0]
    fn drop(&mut self) {}
}

struct StructNoDrop {
    x: i32
}

//~ TRANS_ITEM fn core::ptr[0]::drop_in_place[0]<non_generic_drop_glue::EnumWithDrop[0]> @@ non_generic_drop_glue0[Internal]
enum EnumWithDrop {
    A(i32)
}

impl Drop for EnumWithDrop {
    //~ TRANS_ITEM fn non_generic_drop_glue::{{impl}}[1]::drop[0]
    fn drop(&mut self) {}
}

enum EnumNoDrop {
    A(i32)
}

//~ TRANS_ITEM fn alloc::allocator[0]::{{impl}}[0]::align[0] @@ non_generic_drop_glue0[Internal]
//~ TRANS_ITEM fn alloc::allocator[0]::{{impl}}[0]::from_size_align_unchecked[0] @@ non_generic_drop_glue0[Internal]
//~ TRANS_ITEM fn alloc::allocator[0]::{{impl}}[0]::size[0] @@ non_generic_drop_glue0[Internal]
//~ TRANS_ITEM fn alloc::heap[0]::box_free[0]<core::any[0]::Any[0]> @@ non_generic_drop_glue0[Internal]
//~ TRANS_ITEM fn alloc::heap[0]::{{impl}}[0]::dealloc[0] @@ non_generic_drop_glue0[Internal]
//~ TRANS_ITEM fn core::mem[0]::uninitialized[0]<std::rt[0]::lang_start[0]::{{closure}}[0]<()>> @@ non_generic_drop_glue0[Internal]
//~ TRANS_ITEM fn core::ptr[0]::drop_in_place[0]<alloc::boxed[0]::Box[0]<core::any[0]::Any[0]>> @@ non_generic_drop_glue0[Internal]
//~ TRANS_ITEM fn core::ptr[0]::drop_in_place[0]<core::any[0]::Any[0]> @@ non_generic_drop_glue0[Internal]
//~ TRANS_ITEM fn core::ptr[0]::drop_in_place[0]<core::result[0]::Result[0]<i32, alloc::boxed[0]::Box[0]<core::any[0]::Any[0]>>> @@ non_generic_drop_glue0[Internal]
//~ TRANS_ITEM fn core::ptr[0]::read[0]<std::rt[0]::lang_start[0]::{{closure}}[0]<()>> @@ non_generic_drop_glue0[Internal]
//~ TRANS_ITEM fn core::ptr[0]::write[0]<i32> @@ non_generic_drop_glue0[Internal]
//~ TRANS_ITEM fn core::result[0]::{{impl}}[0]::unwrap_or[0]<i32, alloc::boxed[0]::Box[0]<core::any[0]::Any[0]>> @@ non_generic_drop_glue0[Internal]
//~ TRANS_ITEM fn std::panic[0]::catch_unwind[0]<std::rt[0]::lang_start[0]::{{closure}}[0]<()>, i32> @@ non_generic_drop_glue0[Internal]
//~ TRANS_ITEM fn std::panicking[0]::try[0]::do_call[0]<std::rt[0]::lang_start[0]::{{closure}}[0]<()>, i32> @@ non_generic_drop_glue0[Internal]
//~ TRANS_ITEM fn std::panicking[0]::try[0]<i32, std::rt[0]::lang_start[0]::{{closure}}[0]<()>> @@ non_generic_drop_glue0[Internal]
//~ TRANS_ITEM fn std::rt[0]::lang_start[0]::{{closure}}[0]::{{closure}}[0]<(), i32, extern "rust-call" fn(()) -> i32, fn()> @@ non_generic_drop_glue0[Internal]
//~ TRANS_ITEM fn std::rt[0]::lang_start[0]::{{closure}}[0]<(), i32, extern "rust-call" fn(()) -> i32, &fn()> @@ non_generic_drop_glue0[Internal]
//~ TRANS_ITEM fn std::rt[0]::lang_start[0]<()> @@ non_generic_drop_glue0[External]
//~ TRANS_ITEM fn std::sys_common[0]::backtrace[0]::__rust_begin_short_backtrace[0]<std::rt[0]::lang_start[0]::{{closure}}[0]::{{closure}}[0]<()>, i32> @@ non_generic_drop_glue0[Internal]
//~ TRANS_ITEM fn non_generic_drop_glue::main[0]
fn main() {
    let _ = StructWithDrop { x: 0 }.x;
    let _ = StructNoDrop { x: 0 }.x;
    let _ = match EnumWithDrop::A(0) {
        EnumWithDrop::A(x) => x
    };
    let _ = match EnumNoDrop::A(0) {
        EnumNoDrop::A(x) => x
    };
}
