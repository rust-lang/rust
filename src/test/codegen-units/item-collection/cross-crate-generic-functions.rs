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

// aux-build:cgu_generic_function.rs
extern crate cgu_generic_function;

//~ TRANS_ITEM fn cross_crate_generic_functions::main[0]
fn main()
{
    //~ TRANS_ITEM fn cgu_generic_function::bar[0]<u32>
    //~ TRANS_ITEM fn cgu_generic_function::foo[0]<u32>
    let _ = cgu_generic_function::foo(1u32);

    //~ TRANS_ITEM fn cgu_generic_function::bar[0]<u64>
    //~ TRANS_ITEM fn cgu_generic_function::foo[0]<u64>
    let _ = cgu_generic_function::foo(2u64);

    // This should not introduce a codegen item
    let _ = cgu_generic_function::exported_but_not_generic(3);
}

//~ TRANS_ITEM drop-glue i8
