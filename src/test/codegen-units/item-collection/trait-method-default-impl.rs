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

trait SomeTrait {
    fn foo(&self) { }
    fn bar<T>(&self, x: T) -> T { x }
}

impl SomeTrait for i8 {
    // take the default implementations

    // For the non-generic foo(), we should generate a codegen-item even if it
    // is not called anywhere
    //~ TRANS_ITEM fn trait_method_default_impl::SomeTrait[0]::foo[0]<i8>
}

trait SomeGenericTrait<T1> {
    fn foo(&self) { }
    fn bar<T2>(&self, x: T1, y: T2) {}
}

// Non-generic impl of generic trait
impl SomeGenericTrait<u64> for i32 {
    // take the default implementations

    // For the non-generic foo(), we should generate a codegen-item even if it
    // is not called anywhere
    //~ TRANS_ITEM fn trait_method_default_impl::SomeGenericTrait[0]::foo[0]<i32, u64>
}

// Non-generic impl of generic trait
impl<T1> SomeGenericTrait<T1> for u32 {
    // take the default implementations
    // since nothing is monomorphic here, nothing should be generated unless used somewhere.
}

//~ TRANS_ITEM fn alloc::allocator[0]::{{impl}}[0]::align[0] @@ trait_method_default_impl0[Internal]
//~ TRANS_ITEM fn alloc::allocator[0]::{{impl}}[0]::from_size_align_unchecked[0] @@ trait_method_default_impl0[Internal]
//~ TRANS_ITEM fn alloc::allocator[0]::{{impl}}[0]::size[0] @@ trait_method_default_impl0[Internal]
//~ TRANS_ITEM fn alloc::heap[0]::box_free[0]<core::any[0]::Any[0]> @@ trait_method_default_impl0[Internal]
//~ TRANS_ITEM fn alloc::heap[0]::{{impl}}[0]::dealloc[0] @@ trait_method_default_impl0[Internal]
//~ TRANS_ITEM fn core::mem[0]::uninitialized[0]<std::rt[0]::lang_start[0]::{{closure}}[0]<()>> @@ trait_method_default_impl0[Internal]
//~ TRANS_ITEM fn core::ptr[0]::drop_in_place[0]<alloc::boxed[0]::Box[0]<core::any[0]::Any[0]>> @@ trait_method_default_impl0[Internal]
//~ TRANS_ITEM fn core::ptr[0]::drop_in_place[0]<core::any[0]::Any[0]> @@ trait_method_default_impl0[Internal]
//~ TRANS_ITEM fn core::ptr[0]::drop_in_place[0]<core::result[0]::Result[0]<i32, alloc::boxed[0]::Box[0]<core::any[0]::Any[0]>>> @@ trait_method_default_impl0[Internal]
//~ TRANS_ITEM fn core::ptr[0]::read[0]<std::rt[0]::lang_start[0]::{{closure}}[0]<()>> @@ trait_method_default_impl0[Internal]
//~ TRANS_ITEM fn core::ptr[0]::write[0]<i32> @@ trait_method_default_impl0[Internal]
//~ TRANS_ITEM fn core::result[0]::{{impl}}[0]::unwrap_or[0]<i32, alloc::boxed[0]::Box[0]<core::any[0]::Any[0]>> @@ trait_method_default_impl0[Internal]
//~ TRANS_ITEM fn std::panic[0]::catch_unwind[0]<std::rt[0]::lang_start[0]::{{closure}}[0]<()>, i32> @@ trait_method_default_impl0[Internal]
//~ TRANS_ITEM fn std::panicking[0]::try[0]::do_call[0]<std::rt[0]::lang_start[0]::{{closure}}[0]<()>, i32> @@ trait_method_default_impl0[Internal]
//~ TRANS_ITEM fn std::panicking[0]::try[0]<i32, std::rt[0]::lang_start[0]::{{closure}}[0]<()>> @@ trait_method_default_impl0[Internal]
//~ TRANS_ITEM fn std::rt[0]::lang_start[0]::{{closure}}[0]::{{closure}}[0]<(), i32, extern "rust-call" fn(()) -> i32, fn()> @@ trait_method_default_impl0[Internal]
//~ TRANS_ITEM fn std::rt[0]::lang_start[0]::{{closure}}[0]<(), i32, extern "rust-call" fn(()) -> i32, &fn()> @@ trait_method_default_impl0[Internal]
//~ TRANS_ITEM fn std::rt[0]::lang_start[0]<()> @@ trait_method_default_impl0[External]
//~ TRANS_ITEM fn std::sys_common[0]::backtrace[0]::__rust_begin_short_backtrace[0]<std::rt[0]::lang_start[0]::{{closure}}[0]::{{closure}}[0]<()>, i32> @@ trait_method_default_impl0[Internal]
//~ TRANS_ITEM fn trait_method_default_impl::main[0]
fn main() {
    //~ TRANS_ITEM fn trait_method_default_impl::SomeTrait[0]::bar[0]<i8, char>
    let _ = 1i8.bar('c');

    //~ TRANS_ITEM fn trait_method_default_impl::SomeTrait[0]::bar[0]<i8, &str>
    let _ = 2i8.bar("&str");

    //~ TRANS_ITEM fn trait_method_default_impl::SomeGenericTrait[0]::bar[0]<i32, u64, char>
    0i32.bar(0u64, 'c');

    //~ TRANS_ITEM fn trait_method_default_impl::SomeGenericTrait[0]::bar[0]<i32, u64, &str>
    0i32.bar(0u64, "&str");

    //~ TRANS_ITEM fn trait_method_default_impl::SomeGenericTrait[0]::bar[0]<u32, i8, &[char; 1]>
    0u32.bar(0i8, &['c']);

    //~ TRANS_ITEM fn trait_method_default_impl::SomeGenericTrait[0]::bar[0]<u32, i16, ()>
    0u32.bar(0i16, ());
}
