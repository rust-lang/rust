// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

const RHS: u8 = 8;
const IRHS: i8 = 8;
const RHS16: u16 = 8;
const IRHS16: i16 = 8;
const RHS32: u32 = 8;
const IRHS32: i32 = 8;
const RHS64: u64 = 8;
const IRHS64: i64 = 8;
const RHSUS: usize = 8;
const IRHSIS: isize = 8;

fn main() {
    let _: [&'static str; 1 << RHS] = [""; 256];
    let _: [&'static str; 1 << IRHS] = [""; 256];
    let _: [&'static str; 1 << RHS16] = [""; 256];
    let _: [&'static str; 1 << IRHS16] = [""; 256];
    let _: [&'static str; 1 << RHS32] = [""; 256];
    let _: [&'static str; 1 << IRHS32] = [""; 256];
    let _: [&'static str; 1 << RHS64] = [""; 256];
    let _: [&'static str; 1 << IRHS64] = [""; 256];
    let _: [&'static str; 1 << RHSUS] = [""; 256];
    let _: [&'static str; 1 << IRHSIS] = [""; 256];
}
