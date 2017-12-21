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

trait Trait : Sized {
    fn foo(self) -> Self { self }
}

impl Trait for u32 {
    fn foo(self) -> u32 { self }
}

impl Trait for char {
}

fn take_foo_once<T, F: FnOnce(T) -> T>(f: F, arg: T) -> T {
    (f)(arg)
}

fn take_foo<T, F: Fn(T) -> T>(f: F, arg: T) -> T {
    (f)(arg)
}

fn take_foo_mut<T, F: FnMut(T) -> T>(mut f: F, arg: T) -> T {
    (f)(arg)
}

//~ TRANS_ITEM fn alloc::allocator[0]::{{impl}}[0]::align[0] @@ trait_method_as_argument0[Internal]
//~ TRANS_ITEM fn alloc::allocator[0]::{{impl}}[0]::from_size_align_unchecked[0] @@ trait_method_as_argument0[Internal]
//~ TRANS_ITEM fn alloc::allocator[0]::{{impl}}[0]::size[0] @@ trait_method_as_argument0[Internal]
//~ TRANS_ITEM fn alloc::heap[0]::box_free[0]<core::any[0]::Any[0]> @@ trait_method_as_argument0[Internal]
//~ TRANS_ITEM fn alloc::heap[0]::{{impl}}[0]::dealloc[0] @@ trait_method_as_argument0[Internal]
//~ TRANS_ITEM fn core::mem[0]::uninitialized[0]<std::rt[0]::lang_start[0]::{{closure}}[0]<()>> @@ trait_method_as_argument0[Internal]
//~ TRANS_ITEM fn core::ptr[0]::drop_in_place[0]<alloc::boxed[0]::Box[0]<core::any[0]::Any[0]>> @@ trait_method_as_argument0[Internal]
//~ TRANS_ITEM fn core::ptr[0]::drop_in_place[0]<core::any[0]::Any[0]> @@ trait_method_as_argument0[Internal]
//~ TRANS_ITEM fn core::ptr[0]::drop_in_place[0]<core::result[0]::Result[0]<i32, alloc::boxed[0]::Box[0]<core::any[0]::Any[0]>>> @@ trait_method_as_argument0[Internal]
//~ TRANS_ITEM fn core::ptr[0]::read[0]<std::rt[0]::lang_start[0]::{{closure}}[0]<()>> @@ trait_method_as_argument0[Internal]
//~ TRANS_ITEM fn core::ptr[0]::write[0]<i32> @@ trait_method_as_argument0[Internal]
//~ TRANS_ITEM fn core::result[0]::{{impl}}[0]::unwrap_or[0]<i32, alloc::boxed[0]::Box[0]<core::any[0]::Any[0]>> @@ trait_method_as_argument0[Internal]
//~ TRANS_ITEM fn std::panic[0]::catch_unwind[0]<std::rt[0]::lang_start[0]::{{closure}}[0]<()>, i32> @@ trait_method_as_argument0[Internal]
//~ TRANS_ITEM fn std::panicking[0]::try[0]::do_call[0]<std::rt[0]::lang_start[0]::{{closure}}[0]<()>, i32> @@ trait_method_as_argument0[Internal]
//~ TRANS_ITEM fn std::panicking[0]::try[0]<i32, std::rt[0]::lang_start[0]::{{closure}}[0]<()>> @@ trait_method_as_argument0[Internal]
//~ TRANS_ITEM fn std::rt[0]::lang_start[0]::{{closure}}[0]::{{closure}}[0]<(), i32, extern "rust-call" fn(()) -> i32, fn()> @@ trait_method_as_argument0[Internal]
//~ TRANS_ITEM fn std::rt[0]::lang_start[0]::{{closure}}[0]<(), i32, extern "rust-call" fn(()) -> i32, &fn()> @@ trait_method_as_argument0[Internal]
//~ TRANS_ITEM fn std::rt[0]::lang_start[0]<()> @@ trait_method_as_argument0[External]
//~ TRANS_ITEM fn std::sys_common[0]::backtrace[0]::__rust_begin_short_backtrace[0]<std::rt[0]::lang_start[0]::{{closure}}[0]::{{closure}}[0]<()>, i32> @@ trait_method_as_argument0[Internal]
//~ TRANS_ITEM fn trait_method_as_argument::main[0]
fn main() {
    //~ TRANS_ITEM fn trait_method_as_argument::take_foo_once[0]<u32, fn(u32) -> u32>
    //~ TRANS_ITEM fn trait_method_as_argument::{{impl}}[0]::foo[0]
    //~ TRANS_ITEM fn core::ops[0]::function[0]::FnOnce[0]::call_once[0]<fn(u32) -> u32, (u32)>
    take_foo_once(Trait::foo, 0u32);

    //~ TRANS_ITEM fn trait_method_as_argument::take_foo_once[0]<char, fn(char) -> char>
    //~ TRANS_ITEM fn trait_method_as_argument::Trait[0]::foo[0]<char>
    //~ TRANS_ITEM fn core::ops[0]::function[0]::FnOnce[0]::call_once[0]<fn(char) -> char, (char)>
    take_foo_once(Trait::foo, 'c');

    //~ TRANS_ITEM fn trait_method_as_argument::take_foo[0]<u32, fn(u32) -> u32>
    //~ TRANS_ITEM fn core::ops[0]::function[0]::Fn[0]::call[0]<fn(u32) -> u32, (u32)>
    take_foo(Trait::foo, 0u32);

    //~ TRANS_ITEM fn trait_method_as_argument::take_foo[0]<char, fn(char) -> char>
    //~ TRANS_ITEM fn core::ops[0]::function[0]::Fn[0]::call[0]<fn(char) -> char, (char)>
    take_foo(Trait::foo, 'c');

    //~ TRANS_ITEM fn trait_method_as_argument::take_foo_mut[0]<u32, fn(u32) -> u32>
    //~ TRANS_ITEM fn core::ops[0]::function[0]::FnMut[0]::call_mut[0]<fn(char) -> char, (char)>
    take_foo_mut(Trait::foo, 0u32);

    //~ TRANS_ITEM fn trait_method_as_argument::take_foo_mut[0]<char, fn(char) -> char>
    //~ TRANS_ITEM fn core::ops[0]::function[0]::FnMut[0]::call_mut[0]<fn(u32) -> u32, (u32)>
    take_foo_mut(Trait::foo, 'c');
}
