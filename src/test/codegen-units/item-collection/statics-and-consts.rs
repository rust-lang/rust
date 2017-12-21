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

static STATIC1: i64 = {
    const STATIC1_CONST1: i64 = 2;
    1 + CONST1 as i64 + STATIC1_CONST1
};

const CONST1: i64 = {
    const CONST1_1: i64 = {
        const CONST1_1_1: i64 = 2;
        CONST1_1_1 + 1
    };
    1 + CONST1_1 as i64
};

fn foo() {
    let _ = {
        const CONST2: i64 = 0;
        static STATIC2: i64 = CONST2;

        let x = {
            const CONST2: i64 = 1;
            static STATIC2: i64 = CONST2;
            STATIC2
        };

        x + STATIC2
    };

    let _ = {
        const CONST2: i64 = 0;
        static STATIC2: i64 = CONST2;
        STATIC2
    };
}

fn main() {
    foo();
    let _ = STATIC1;
}

//~ TRANS_ITEM static statics_and_consts::STATIC1[0]

//~ TRANS_ITEM fn statics_and_consts::foo[0]
//~ TRANS_ITEM static statics_and_consts::foo[0]::STATIC2[0]
//~ TRANS_ITEM static statics_and_consts::foo[0]::STATIC2[1]
//~ TRANS_ITEM static statics_and_consts::foo[0]::STATIC2[2]

//~ TRANS_ITEM fn alloc::allocator[0]::{{impl}}[0]::align[0] @@ statics_and_consts0[Internal]
//~ TRANS_ITEM fn alloc::allocator[0]::{{impl}}[0]::from_size_align_unchecked[0] @@ statics_and_consts0[Internal]
//~ TRANS_ITEM fn alloc::allocator[0]::{{impl}}[0]::size[0] @@ statics_and_consts0[Internal]
//~ TRANS_ITEM fn alloc::heap[0]::box_free[0]<core::any[0]::Any[0]> @@ statics_and_consts0[Internal]
//~ TRANS_ITEM fn alloc::heap[0]::{{impl}}[0]::dealloc[0] @@ statics_and_consts0[Internal]
//~ TRANS_ITEM fn core::mem[0]::uninitialized[0]<std::rt[0]::lang_start[0]::{{closure}}[0]<()>> @@ statics_and_consts0[Internal]
//~ TRANS_ITEM fn core::ptr[0]::drop_in_place[0]<alloc::boxed[0]::Box[0]<core::any[0]::Any[0]>> @@ statics_and_consts0[Internal]
//~ TRANS_ITEM fn core::ptr[0]::drop_in_place[0]<core::any[0]::Any[0]> @@ statics_and_consts0[Internal]
//~ TRANS_ITEM fn core::ptr[0]::drop_in_place[0]<core::result[0]::Result[0]<i32, alloc::boxed[0]::Box[0]<core::any[0]::Any[0]>>> @@ statics_and_consts0[Internal]
//~ TRANS_ITEM fn core::ptr[0]::read[0]<std::rt[0]::lang_start[0]::{{closure}}[0]<()>> @@ statics_and_consts0[Internal]
//~ TRANS_ITEM fn core::ptr[0]::write[0]<i32> @@ statics_and_consts0[Internal]
//~ TRANS_ITEM fn core::result[0]::{{impl}}[0]::unwrap_or[0]<i32, alloc::boxed[0]::Box[0]<core::any[0]::Any[0]>> @@ statics_and_consts0[Internal]
//~ TRANS_ITEM fn std::panic[0]::catch_unwind[0]<std::rt[0]::lang_start[0]::{{closure}}[0]<()>, i32> @@ statics_and_consts0[Internal]
//~ TRANS_ITEM fn std::panicking[0]::try[0]::do_call[0]<std::rt[0]::lang_start[0]::{{closure}}[0]<()>, i32> @@ statics_and_consts0[Internal]
//~ TRANS_ITEM fn std::panicking[0]::try[0]<i32, std::rt[0]::lang_start[0]::{{closure}}[0]<()>> @@ statics_and_consts0[Internal]
//~ TRANS_ITEM fn std::rt[0]::lang_start[0]::{{closure}}[0]::{{closure}}[0]<(), i32, extern "rust-call" fn(()) -> i32, fn()> @@ statics_and_consts0[Internal]
//~ TRANS_ITEM fn std::rt[0]::lang_start[0]::{{closure}}[0]<(), i32, extern "rust-call" fn(()) -> i32, &fn()> @@ statics_and_consts0[Internal]
//~ TRANS_ITEM fn std::rt[0]::lang_start[0]<()> @@ statics_and_consts0[External]
//~ TRANS_ITEM fn std::sys_common[0]::backtrace[0]::__rust_begin_short_backtrace[0]<std::rt[0]::lang_start[0]::{{closure}}[0]::{{closure}}[0]<()>, i32> @@ statics_and_consts0[Internal]
//~ TRANS_ITEM fn statics_and_consts::main[0]
