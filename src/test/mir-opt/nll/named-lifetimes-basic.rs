// Copyright 2012-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Basic test for named lifetime translation. Check that we
// instantiate the types that appear in function arguments with
// suitable variables and that we setup the outlives relationship
// between R0 and R1 properly.

// compile-flags:-Znll -Zverbose
//                     ^^^^^^^^^ force compiler to dump more region information
// ignore-tidy-linelength

#![allow(warnings)]

fn use_x<'a, 'b: 'a, 'c>(w: &'a mut i32, x: &'b u32, y: &'a u32, z: &'c u32) -> bool { true }

fn main() {
}

// END RUST SOURCE
// START rustc.use_x.nll.0.mir
// | Free Region Mapping
// | '_#0r    | Global   | ['_#2r, '_#1r, '_#0r, '_#4r, '_#3r]
// | '_#1r    | External | ['_#1r, '_#4r]
// | '_#2r    | External | ['_#2r, '_#1r, '_#4r]
// | '_#3r    | Local    | ['_#4r, '_#3r]
// | '_#4r    | Local    | ['_#4r]
// |
// | Inferred Region Values
// | '_#0r    | {'_#0r, bb0[0..=1]}
// | '_#1r    | {'_#1r, bb0[0..=1]}
// | '_#2r    | {'_#2r, bb0[0..=1]}
// | '_#3r    | {'_#3r, bb0[0..=1]}
// | '_#4r    | {'_#4r, bb0[0..=1]}
// | '_#5r    | {'_#1r, bb0[0..=1]}
// | '_#6r    | {'_#2r, bb0[0..=1]}
// | '_#7r    | {'_#1r, bb0[0..=1]}
// | '_#8r    | {'_#3r, bb0[0..=1]}
// |
// ...
// fn use_x(_1: &'_#5r mut i32, _2: &'_#6r u32, _3: &'_#7r u32, _4: &'_#8r u32) -> bool {
// END rustc.use_x.nll.0.mir
