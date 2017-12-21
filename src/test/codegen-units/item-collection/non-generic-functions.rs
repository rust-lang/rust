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

//~ TRANS_ITEM fn non_generic_functions::foo[0]
fn foo() {
    {
        //~ TRANS_ITEM fn non_generic_functions::foo[0]::foo[0]
        fn foo() {}
        foo();
    }

    {
        //~ TRANS_ITEM fn non_generic_functions::foo[0]::foo[1]
        fn foo() {}
        foo();
    }
}

//~ TRANS_ITEM fn non_generic_functions::bar[0]
fn bar() {
    //~ TRANS_ITEM fn non_generic_functions::bar[0]::baz[0]
    fn baz() {}
    baz();
}

struct Struct { _x: i32 }

impl Struct {
    //~ TRANS_ITEM fn non_generic_functions::{{impl}}[0]::foo[0]
    fn foo() {
        {
            //~ TRANS_ITEM fn non_generic_functions::{{impl}}[0]::foo[0]::foo[0]
            fn foo() {}
            foo();
        }

        {
            //~ TRANS_ITEM fn non_generic_functions::{{impl}}[0]::foo[0]::foo[1]
            fn foo() {}
            foo();
        }
    }

    //~ TRANS_ITEM fn non_generic_functions::{{impl}}[0]::bar[0]
    fn bar(&self) {
        {
            //~ TRANS_ITEM fn non_generic_functions::{{impl}}[0]::bar[0]::foo[0]
            fn foo() {}
            foo();
        }

        {
            //~ TRANS_ITEM fn non_generic_functions::{{impl}}[0]::bar[0]::foo[1]
            fn foo() {}
            foo();
        }
    }
}

//~ TRANS_ITEM fn alloc::allocator[0]::{{impl}}[0]::align[0] @@ non_generic_functions0[Internal]
//~ TRANS_ITEM fn alloc::allocator[0]::{{impl}}[0]::from_size_align_unchecked[0] @@ non_generic_functions0[Internal]
//~ TRANS_ITEM fn alloc::allocator[0]::{{impl}}[0]::size[0] @@ non_generic_functions0[Internal]
//~ TRANS_ITEM fn alloc::heap[0]::box_free[0]<core::any[0]::Any[0]> @@ non_generic_functions0[Internal]
//~ TRANS_ITEM fn alloc::heap[0]::{{impl}}[0]::dealloc[0] @@ non_generic_functions0[Internal]
//~ TRANS_ITEM fn core::mem[0]::uninitialized[0]<std::rt[0]::lang_start[0]::{{closure}}[0]<()>> @@ non_generic_functions0[Internal]
//~ TRANS_ITEM fn core::ptr[0]::drop_in_place[0]<alloc::boxed[0]::Box[0]<core::any[0]::Any[0]>> @@ non_generic_functions0[Internal]
//~ TRANS_ITEM fn core::ptr[0]::drop_in_place[0]<core::any[0]::Any[0]> @@ non_generic_functions0[Internal]
//~ TRANS_ITEM fn core::ptr[0]::drop_in_place[0]<core::result[0]::Result[0]<i32, alloc::boxed[0]::Box[0]<core::any[0]::Any[0]>>> @@ non_generic_functions0[Internal]
//~ TRANS_ITEM fn core::ptr[0]::read[0]<std::rt[0]::lang_start[0]::{{closure}}[0]<()>> @@ non_generic_functions0[Internal]
//~ TRANS_ITEM fn core::ptr[0]::write[0]<i32> @@ non_generic_functions0[Internal]
//~ TRANS_ITEM fn core::result[0]::{{impl}}[0]::unwrap_or[0]<i32, alloc::boxed[0]::Box[0]<core::any[0]::Any[0]>> @@ non_generic_functions0[Internal]
//~ TRANS_ITEM fn std::panic[0]::catch_unwind[0]<std::rt[0]::lang_start[0]::{{closure}}[0]<()>, i32> @@ non_generic_functions0[Internal]
//~ TRANS_ITEM fn std::panicking[0]::try[0]::do_call[0]<std::rt[0]::lang_start[0]::{{closure}}[0]<()>, i32> @@ non_generic_functions0[Internal]
//~ TRANS_ITEM fn std::panicking[0]::try[0]<i32, std::rt[0]::lang_start[0]::{{closure}}[0]<()>> @@ non_generic_functions0[Internal]
//~ TRANS_ITEM fn std::rt[0]::lang_start[0]::{{closure}}[0]::{{closure}}[0]<(), i32, extern "rust-call" fn(()) -> i32, fn()> @@ non_generic_functions0[Internal]
//~ TRANS_ITEM fn std::rt[0]::lang_start[0]::{{closure}}[0]<(), i32, extern "rust-call" fn(()) -> i32, &fn()> @@ non_generic_functions0[Internal]
//~ TRANS_ITEM fn std::rt[0]::lang_start[0]<()> @@ non_generic_functions0[External]
//~ TRANS_ITEM fn std::sys_common[0]::backtrace[0]::__rust_begin_short_backtrace[0]<std::rt[0]::lang_start[0]::{{closure}}[0]::{{closure}}[0]<()>, i32> @@ non_generic_functions0[Internal]
//~ TRANS_ITEM fn non_generic_functions::main[0]
fn main() {
    foo();
    bar();
    Struct::foo();
    let x = Struct { _x: 0 };
    x.bar();
}
