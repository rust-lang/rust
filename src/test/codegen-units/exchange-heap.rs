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

//~ TRANS_ITEM fn exchange_heap::main[0]
fn main() {

    //~ TRANS_ITEM drop-glue Box<i32>
    //~ TRANS_ITEM fn alloc[0]::boxed[0]::Box<T>[0]::new[0]<i32>
    //~ TRANS_ITEM fn alloc[0]::heap[0]::exchange_malloc[0]
    //~ TRANS_ITEM fn alloc[0]::heap[0]::exchange_free[0]
    let _x = Box::new(0i32);

    // We also get a codegen item for alloc::heap::deallocate() because it's
    // called from alloc::heap::exchange_free() and is marked with #[inline]:
    //~ TRANS_ITEM fn alloc[0]::heap[0]::deallocate[0]

    // And these we get from alloc::heap::exchange_malloc:
    //~ TRANS_ITEM fn alloc[0]::heap[0]::allocate[0]
    //~ TRANS_ITEM fn alloc[0]::heap[0]::check_size_and_alignment[0]
    //~ TRANS_ITEM fn core[0]::ptr[0]::*mut T[0]::is_null[0]<u8>
    //~ TRANS_ITEM fn core[0]::ptr[0]::null_mut[0]<u8>
    //~ TRANS_ITEM fn core[0]::fmt[0]::ArgumentV1<'a>[0]::new[0]<usize>
    //~ TRANS_ITEM fn core[0]::fmt[0]::Arguments<'a>[0]::new_v1[0]
    //~ TRANS_ITEM fn core[0]::num[0]::usize.One[0]::one[0]
    //~ TRANS_ITEM fn core[0]::num[0]::usize.Zero[0]::zero[0]
    //~ TRANS_ITEM fn core[0]::num[0]::usize[0]::is_power_of_two[0]
    //~ TRANS_ITEM fn core[0]::num[0]::usize[0]::wrapping_sub[0]
}

//~ TRANS_ITEM drop-glue i8
