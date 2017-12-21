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

struct Struct<T> {
    x: T,
    f: fn(x: T) -> T,
}

fn id<T>(x: T) -> T { x }

impl<T> Struct<T> {

    fn new(x: T) -> Struct<T> {
        Struct {
            x: x,
            f: id
        }
    }

    fn get<T2>(self, x: T2) -> (T, T2) {
        (self.x, x)
    }
}

pub struct LifeTimeOnly<'a> {
    _a: &'a u32
}

impl<'a> LifeTimeOnly<'a> {

    //~ TRANS_ITEM fn generic_impl::{{impl}}[1]::foo[0]
    pub fn foo(&self) {}
    //~ TRANS_ITEM fn generic_impl::{{impl}}[1]::bar[0]
    pub fn bar(&'a self) {}
    //~ TRANS_ITEM fn generic_impl::{{impl}}[1]::baz[0]
    pub fn baz<'b>(&'b self) {}

    pub fn non_instantiated<T>(&self) {}
}

//~ TRANS_ITEM fn alloc::allocator[0]::{{impl}}[0]::align[0] @@ generic_impl0[Internal]
//~ TRANS_ITEM fn alloc::allocator[0]::{{impl}}[0]::from_size_align_unchecked[0] @@ generic_impl0[Internal]
//~ TRANS_ITEM fn alloc::allocator[0]::{{impl}}[0]::size[0] @@ generic_impl0[Internal]
//~ TRANS_ITEM fn alloc::heap[0]::box_free[0]<core::any[0]::Any[0]> @@ generic_impl0[Internal]
//~ TRANS_ITEM fn alloc::heap[0]::{{impl}}[0]::dealloc[0] @@ generic_impl0[Internal]
//~ TRANS_ITEM fn core::mem[0]::uninitialized[0]<std::rt[0]::lang_start[0]::{{closure}}[0]<()>> @@ generic_impl0[Internal]
//~ TRANS_ITEM fn core::ptr[0]::drop_in_place[0]<alloc::boxed[0]::Box[0]<core::any[0]::Any[0]>> @@ generic_impl0[Internal]
//~ TRANS_ITEM fn core::ptr[0]::drop_in_place[0]<core::any[0]::Any[0]> @@ generic_impl0[Internal]
//~ TRANS_ITEM fn core::ptr[0]::drop_in_place[0]<core::result[0]::Result[0]<i32, alloc::boxed[0]::Box[0]<core::any[0]::Any[0]>>> @@ generic_impl0[Internal]
//~ TRANS_ITEM fn core::ptr[0]::read[0]<std::rt[0]::lang_start[0]::{{closure}}[0]<()>> @@ generic_impl0[Internal]
//~ TRANS_ITEM fn core::ptr[0]::write[0]<i32> @@ generic_impl0[Internal]
//~ TRANS_ITEM fn core::result[0]::{{impl}}[0]::unwrap_or[0]<i32, alloc::boxed[0]::Box[0]<core::any[0]::Any[0]>> @@ generic_impl0[Internal]
//~ TRANS_ITEM fn std::panic[0]::catch_unwind[0]<std::rt[0]::lang_start[0]::{{closure}}[0]<()>, i32> @@ generic_impl0[Internal]
//~ TRANS_ITEM fn std::panicking[0]::try[0]::do_call[0]<std::rt[0]::lang_start[0]::{{closure}}[0]<()>, i32> @@ generic_impl0[Internal]
//~ TRANS_ITEM fn std::panicking[0]::try[0]<i32, std::rt[0]::lang_start[0]::{{closure}}[0]<()>> @@ generic_impl0[Internal]
//~ TRANS_ITEM fn std::rt[0]::lang_start[0]::{{closure}}[0]::{{closure}}[0]<(), i32, extern "rust-call" fn(()) -> i32, fn()> @@ generic_impl0[Internal]
//~ TRANS_ITEM fn std::rt[0]::lang_start[0]::{{closure}}[0]<(), i32, extern "rust-call" fn(()) -> i32, &fn()> @@ generic_impl0[Internal]
//~ TRANS_ITEM fn std::rt[0]::lang_start[0]<()> @@ generic_impl0[External]
//~ TRANS_ITEM fn std::sys_common[0]::backtrace[0]::__rust_begin_short_backtrace[0]<std::rt[0]::lang_start[0]::{{closure}}[0]::{{closure}}[0]<()>, i32> @@ generic_impl0[Internal]
//~ TRANS_ITEM fn generic_impl::main[0]
fn main() {
    //~ TRANS_ITEM fn generic_impl::{{impl}}[0]::new[0]<i32>
    //~ TRANS_ITEM fn generic_impl::id[0]<i32>
    //~ TRANS_ITEM fn generic_impl::{{impl}}[0]::get[0]<i32, i16>
    let _ = Struct::new(0i32).get(0i16);

    //~ TRANS_ITEM fn generic_impl::{{impl}}[0]::new[0]<i64>
    //~ TRANS_ITEM fn generic_impl::id[0]<i64>
    //~ TRANS_ITEM fn generic_impl::{{impl}}[0]::get[0]<i64, i16>
    let _ = Struct::new(0i64).get(0i16);

    //~ TRANS_ITEM fn generic_impl::{{impl}}[0]::new[0]<char>
    //~ TRANS_ITEM fn generic_impl::id[0]<char>
    //~ TRANS_ITEM fn generic_impl::{{impl}}[0]::get[0]<char, i16>
    let _ = Struct::new('c').get(0i16);

    //~ TRANS_ITEM fn generic_impl::{{impl}}[0]::new[0]<&str>
    //~ TRANS_ITEM fn generic_impl::id[0]<&str>
    //~ TRANS_ITEM fn generic_impl::{{impl}}[0]::get[0]<generic_impl::Struct[0]<&str>, i16>
    let _ = Struct::new(Struct::new("str")).get(0i16);

    //~ TRANS_ITEM fn generic_impl::{{impl}}[0]::new[0]<generic_impl::Struct[0]<&str>>
    //~ TRANS_ITEM fn generic_impl::id[0]<generic_impl::Struct[0]<&str>>
    let _ = (Struct::new(Struct::new("str")).f)(Struct::new("str"));
}
