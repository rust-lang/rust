// Copyright 2012-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Basic test for reborrow constraints: the region (`R5`) that appears
// in the type of `r_a` must outlive the region (`R7`) that appears in
// the type of `r_b`

// compile-flags:-Znll -Zverbose
//                     ^^^^^^^^^ force compiler to dump more region information

#![allow(warnings)]

fn use_x(_: &mut i32) -> bool { true }

fn main() {
    let mut foo: i32     = 22;
    let r_a: &mut i32 = &mut foo;
    let r_b: &mut i32 = &mut *r_a;
    use_x(r_b);
}

// END RUST SOURCE
// START rustc.main.nll.0.mir
// | '_#6r    | {bb0[6], bb0[7], bb0[8], bb0[9], bb0[10], bb0[11], bb0[12], bb0[13], bb0[14]}
// ...
// | '_#8r    | {bb0[11], bb0[12], bb0[13], bb0[14]}
// ...
// let _2: &'_#6r mut i32;
// ...
// let _4: &'_#8r mut i32;
// END rustc.main.nll.0.mir
