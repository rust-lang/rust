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

pub trait SomeTrait {
    fn foo(&self);
    fn bar<T>(&self, x: T);
}

impl SomeTrait for i64 {

    //~ TRANS_ITEM fn trait_implementations::{{impl}}[0]::foo[0]
    fn foo(&self) {}

    fn bar<T>(&self, _: T) {}
}

impl SomeTrait for i32 {

    //~ TRANS_ITEM fn trait_implementations::{{impl}}[1]::foo[0]
    fn foo(&self) {}

    fn bar<T>(&self, _: T) {}
}

pub trait SomeGenericTrait<T> {
    fn foo(&self, x: T);
    fn bar<T2>(&self, x: T, y: T2);
}

// Concrete impl of generic trait
impl SomeGenericTrait<u32> for f64 {

    //~ TRANS_ITEM fn trait_implementations::{{impl}}[2]::foo[0]
    fn foo(&self, _: u32) {}

    fn bar<T2>(&self, _: u32, _: T2) {}
}

// Generic impl of generic trait
impl<T> SomeGenericTrait<T> for f32 {

    fn foo(&self, _: T) {}
    fn bar<T2>(&self, _: T, _: T2) {}
}

//~ TRANS_ITEM fn alloc::allocator[0]::{{impl}}[0]::align[0] @@ trait_implementations0[Internal]
//~ TRANS_ITEM fn alloc::allocator[0]::{{impl}}[0]::from_size_align_unchecked[0] @@ trait_implementations0[Internal]
//~ TRANS_ITEM fn alloc::allocator[0]::{{impl}}[0]::size[0] @@ trait_implementations0[Internal]
//~ TRANS_ITEM fn alloc::heap[0]::box_free[0]<core::any[0]::Any[0]> @@ trait_implementations0[Internal]
//~ TRANS_ITEM fn alloc::heap[0]::{{impl}}[0]::dealloc[0] @@ trait_implementations0[Internal]
//~ TRANS_ITEM fn core::mem[0]::uninitialized[0]<std::rt[0]::lang_start[0]::{{closure}}[0]<()>> @@ trait_implementations0[Internal]
//~ TRANS_ITEM fn core::ptr[0]::drop_in_place[0]<alloc::boxed[0]::Box[0]<core::any[0]::Any[0]>> @@ trait_implementations0[Internal]
//~ TRANS_ITEM fn core::ptr[0]::drop_in_place[0]<core::any[0]::Any[0]> @@ trait_implementations0[Internal]
//~ TRANS_ITEM fn core::ptr[0]::drop_in_place[0]<core::result[0]::Result[0]<i32, alloc::boxed[0]::Box[0]<core::any[0]::Any[0]>>> @@ trait_implementations0[Internal]
//~ TRANS_ITEM fn core::ptr[0]::read[0]<std::rt[0]::lang_start[0]::{{closure}}[0]<()>> @@ trait_implementations0[Internal]
//~ TRANS_ITEM fn core::ptr[0]::write[0]<i32> @@ trait_implementations0[Internal]
//~ TRANS_ITEM fn core::result[0]::{{impl}}[0]::unwrap_or[0]<i32, alloc::boxed[0]::Box[0]<core::any[0]::Any[0]>> @@ trait_implementations0[Internal]
//~ TRANS_ITEM fn std::panic[0]::catch_unwind[0]<std::rt[0]::lang_start[0]::{{closure}}[0]<()>, i32> @@ trait_implementations0[Internal]
//~ TRANS_ITEM fn std::panicking[0]::try[0]::do_call[0]<std::rt[0]::lang_start[0]::{{closure}}[0]<()>, i32> @@ trait_implementations0[Internal]
//~ TRANS_ITEM fn std::panicking[0]::try[0]<i32, std::rt[0]::lang_start[0]::{{closure}}[0]<()>> @@ trait_implementations0[Internal]
//~ TRANS_ITEM fn std::rt[0]::lang_start[0]::{{closure}}[0]::{{closure}}[0]<(), i32, extern "rust-call" fn(()) -> i32, fn()> @@ trait_implementations0[Internal]
//~ TRANS_ITEM fn std::rt[0]::lang_start[0]::{{closure}}[0]<(), i32, extern "rust-call" fn(()) -> i32, &fn()> @@ trait_implementations0[Internal]
//~ TRANS_ITEM fn std::rt[0]::lang_start[0]<()> @@ trait_implementations0[External]
//~ TRANS_ITEM fn std::sys_common[0]::backtrace[0]::__rust_begin_short_backtrace[0]<std::rt[0]::lang_start[0]::{{closure}}[0]::{{closure}}[0]<()>, i32> @@ trait_implementations0[Internal]
//~ TRANS_ITEM fn trait_implementations::main[0]
fn main() {
   //~ TRANS_ITEM fn trait_implementations::{{impl}}[1]::bar[0]<char>
   0i32.bar('x');

   //~ TRANS_ITEM fn trait_implementations::{{impl}}[2]::bar[0]<&str>
   0f64.bar(0u32, "&str");

   //~ TRANS_ITEM fn trait_implementations::{{impl}}[2]::bar[0]<()>
   0f64.bar(0u32, ());

   //~ TRANS_ITEM fn trait_implementations::{{impl}}[3]::foo[0]<char>
   0f32.foo('x');

   //~ TRANS_ITEM fn trait_implementations::{{impl}}[3]::foo[0]<i64>
   0f32.foo(-1i64);

   //~ TRANS_ITEM fn trait_implementations::{{impl}}[3]::bar[0]<u32, ()>
   0f32.bar(0u32, ());

   //~ TRANS_ITEM fn trait_implementations::{{impl}}[3]::bar[0]<&str, &str>
   0f32.bar("&str", "&str");
}
