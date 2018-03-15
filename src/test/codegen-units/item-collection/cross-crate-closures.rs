// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// In the current version of the collector that still has to support
// legacy-trans, closures do not generate their own TransItems, so we are
// ignoring this test until MIR trans has taken over completely
// ignore-test

// ignore-tidy-linelength
// compile-flags:-Zprint-trans-items=eager

#![deny(dead_code)]

// aux-build:cgu_extern_closures.rs
extern crate cgu_extern_closures;

//~ TRANS_ITEM fn cross_crate_closures::main[0]
fn main() {

    //~ TRANS_ITEM fn cgu_extern_closures::inlined_fn[0]
    //~ TRANS_ITEM fn cgu_extern_closures::inlined_fn[0]::{{closure}}[0]
    let _ = cgu_extern_closures::inlined_fn(1, 2);

    //~ TRANS_ITEM fn cgu_extern_closures::inlined_fn_generic[0]<i32>
    //~ TRANS_ITEM fn cgu_extern_closures::inlined_fn_generic[0]::{{closure}}[0]<i32>
    let _ = cgu_extern_closures::inlined_fn_generic(3, 4, 5i32);

    // Nothing should be generated for this call, we just link to the instance
    // in the extern crate.
    let _ = cgu_extern_closures::non_inlined_fn(6, 7);
}

//~ TRANS_ITEM drop-glue i8
