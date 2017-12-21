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

fn take_fn_once<T1, T2, F: FnOnce(T1, T2)>(f: F, x: T1, y: T2) {
    (f)(x, y)
}

fn function<T1, T2>(_: T1, _: T2) {}

fn take_fn_pointer<T1, T2>(f: fn(T1, T2), x: T1, y: T2) {
    (f)(x, y)
}

//~ TRANS_ITEM fn alloc::allocator[0]::{{impl}}[0]::align[0] @@ function_as_argument0[Internal]
//~ TRANS_ITEM fn alloc::allocator[0]::{{impl}}[0]::from_size_align_unchecked[0] @@ function_as_argument0[Internal]
//~ TRANS_ITEM fn alloc::allocator[0]::{{impl}}[0]::size[0] @@ function_as_argument0[Internal]
//~ TRANS_ITEM fn alloc::heap[0]::box_free[0]<core::any[0]::Any[0]> @@ function_as_argument0[Internal]
//~ TRANS_ITEM fn alloc::heap[0]::{{impl}}[0]::dealloc[0] @@ function_as_argument0[Internal]
//~ TRANS_ITEM fn core::mem[0]::uninitialized[0]<std::rt[0]::lang_start[0]::{{closure}}[0]<()>> @@ function_as_argument0[Internal]
//~ TRANS_ITEM fn core::ptr[0]::drop_in_place[0]<alloc::boxed[0]::Box[0]<core::any[0]::Any[0]>> @@ function_as_argument0[Internal]
//~ TRANS_ITEM fn core::ptr[0]::drop_in_place[0]<core::any[0]::Any[0]> @@ function_as_argument0[Internal]
//~ TRANS_ITEM fn core::ptr[0]::drop_in_place[0]<core::result[0]::Result[0]<i32, alloc::boxed[0]::Box[0]<core::any[0]::Any[0]>>> @@ function_as_argument0[Internal]
//~ TRANS_ITEM fn core::ptr[0]::read[0]<std::rt[0]::lang_start[0]::{{closure}}[0]<()>> @@ function_as_argument0[Internal]
//~ TRANS_ITEM fn core::ptr[0]::write[0]<i32> @@ function_as_argument0[Internal]
//~ TRANS_ITEM fn core::result[0]::{{impl}}[0]::unwrap_or[0]<i32, alloc::boxed[0]::Box[0]<core::any[0]::Any[0]>> @@ function_as_argument0[Internal]
//~ TRANS_ITEM fn std::panic[0]::catch_unwind[0]<std::rt[0]::lang_start[0]::{{closure}}[0]<()>, i32> @@ function_as_argument0[Internal]
//~ TRANS_ITEM fn std::panicking[0]::try[0]::do_call[0]<std::rt[0]::lang_start[0]::{{closure}}[0]<()>, i32> @@ function_as_argument0[Internal]
//~ TRANS_ITEM fn std::panicking[0]::try[0]<i32, std::rt[0]::lang_start[0]::{{closure}}[0]<()>> @@ function_as_argument0[Internal]
//~ TRANS_ITEM fn std::rt[0]::lang_start[0]::{{closure}}[0]::{{closure}}[0]<(), i32, extern "rust-call" fn(()) -> i32, fn()> @@ function_as_argument0[Internal]
//~ TRANS_ITEM fn std::rt[0]::lang_start[0]::{{closure}}[0]<(), i32, extern "rust-call" fn(()) -> i32, &fn()> @@ function_as_argument0[Internal]
//~ TRANS_ITEM fn std::rt[0]::lang_start[0]<()> @@ function_as_argument0[External]
//~ TRANS_ITEM fn std::sys_common[0]::backtrace[0]::__rust_begin_short_backtrace[0]<std::rt[0]::lang_start[0]::{{closure}}[0]::{{closure}}[0]<()>, i32> @@ function_as_argument0[Internal]
//~ TRANS_ITEM fn function_as_argument::main[0]
fn main() {

    //~ TRANS_ITEM fn function_as_argument::take_fn_once[0]<u32, &str, fn(u32, &str)>
    //~ TRANS_ITEM fn function_as_argument::function[0]<u32, &str>
    //~ TRANS_ITEM fn core::ops[0]::function[0]::FnOnce[0]::call_once[0]<fn(u32, &str), (u32, &str)>
    take_fn_once(function, 0u32, "abc");

    //~ TRANS_ITEM fn function_as_argument::take_fn_once[0]<char, f64, fn(char, f64)>
    //~ TRANS_ITEM fn function_as_argument::function[0]<char, f64>
    //~ TRANS_ITEM fn core::ops[0]::function[0]::FnOnce[0]::call_once[0]<fn(char, f64), (char, f64)>
    take_fn_once(function, 'c', 0f64);

    //~ TRANS_ITEM fn function_as_argument::take_fn_pointer[0]<i32, ()>
    //~ TRANS_ITEM fn function_as_argument::function[0]<i32, ()>
    take_fn_pointer(function, 0i32, ());

    //~ TRANS_ITEM fn function_as_argument::take_fn_pointer[0]<f32, i64>
    //~ TRANS_ITEM fn function_as_argument::function[0]<f32, i64>
    take_fn_pointer(function, 0f32, 0i64);
}
