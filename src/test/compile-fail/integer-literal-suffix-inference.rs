// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {

    // the smallest positive values that need these types
    let a8: i8 = 8;
    let a16: i16 = 128;
    let a32: i32 = 32_768;
    let a64: i64 = 2_147_483_648;

    // the smallest negative values that need these types
    let c8: i8 = -9;
    let c16: i16 = -129;
    let c32: i32 = -32_769;
    let c64: i64 = -2_147_483_649;

    fn id_i8(n: i8) -> i8 { n }
    fn id_i16(n: i16) -> i16 { n }
    fn id_i32(n: i32) -> i32 { n }
    fn id_i64(n: i64) -> i64 { n }

    // the smallest values that need these types
    let b8: u8 = 16;
    let b16: u16 = 256;
    let b32: u32 = 65_536;
    let b64: u64 = 4_294_967_296;

    fn id_u8(n: u8) -> u8 { n }
    fn id_u16(n: u16) -> u16 { n }
    fn id_u32(n: u32) -> u32 { n }
    fn id_u64(n: u64) -> u64 { n }

    id_i8(a8); // ok
    id_i8(a16);
    //~^ ERROR mismatched types
    //~| expected i8, found i16
    id_i8(a32);
    //~^ ERROR mismatched types
    //~| expected i8, found i32
    id_i8(a64);
    //~^ ERROR mismatched types
    //~| expected i8, found i64

    id_i16(a8);
    //~^ ERROR mismatched types
    //~| expected i16, found i8
    id_i16(a16); // ok
    id_i16(a32);
    //~^ ERROR mismatched types
    //~| expected i16, found i32
    id_i16(a64);
    //~^ ERROR mismatched types
    //~| expected i16, found i64

    id_i32(a8);
    //~^ ERROR mismatched types
    //~| expected i32, found i8
    id_i32(a16);
    //~^ ERROR mismatched types
    //~| expected i32, found i16
    id_i32(a32); // ok
    id_i32(a64);
    //~^ ERROR mismatched types
    //~| expected i32, found i64

    id_i64(a8);
    //~^ ERROR mismatched types
    //~| expected i64, found i8
    id_i64(a16);
    //~^ ERROR mismatched types
    //~| expected i64, found i16
    id_i64(a32);
    //~^ ERROR mismatched types
    //~| expected i64, found i32
    id_i64(a64); // ok

    id_i8(c8); // ok
    id_i8(c16);
    //~^ ERROR mismatched types
    //~| expected i8, found i16
    id_i8(c32);
    //~^ ERROR mismatched types
    //~| expected i8, found i32
    id_i8(c64);
    //~^ ERROR mismatched types
    //~| expected i8, found i64

    id_i16(c8);
    //~^ ERROR mismatched types
    //~| expected i16, found i8
    id_i16(c16); // ok
    id_i16(c32);
    //~^ ERROR mismatched types
    //~| expected i16, found i32
    id_i16(c64);
    //~^ ERROR mismatched types
    //~| expected i16, found i64

    id_i32(c8);
    //~^ ERROR mismatched types
    //~| expected i32, found i8
    id_i32(c16);
    //~^ ERROR mismatched types
    //~| expected i32, found i16
    id_i32(c32); // ok
    id_i32(c64);
    //~^ ERROR mismatched types
    //~| expected i32, found i64

    id_i64(a8);
    //~^ ERROR mismatched types
    //~| expected i64, found i8
    id_i64(a16);
    //~^ ERROR mismatched types
    //~| expected i64, found i16
    id_i64(a32);
    //~^ ERROR mismatched types
    //~| expected i64, found i32
    id_i64(a64); // ok

    id_u8(b8); // ok
    id_u8(b16);
    //~^ ERROR mismatched types
    //~| expected u8, found u16
    id_u8(b32);
    //~^ ERROR mismatched types
    //~| expected u8, found u32
    id_u8(b64);
    //~^ ERROR mismatched types
    //~| expected u8, found u64

    id_u16(b8);
    //~^ ERROR mismatched types
    //~| expected u16, found u8
    id_u16(b16); // ok
    id_u16(b32);
    //~^ ERROR mismatched types
    //~| expected u16, found u32
    id_u16(b64);
    //~^ ERROR mismatched types
    //~| expected u16, found u64

    id_u32(b8);
    //~^ ERROR mismatched types
    //~| expected u32, found u8
    id_u32(b16);
    //~^ ERROR mismatched types
    //~| expected u32, found u16
    id_u32(b32); // ok
    id_u32(b64);
    //~^ ERROR mismatched types
    //~| expected u32, found u64

    id_u64(b8);
    //~^ ERROR mismatched types
    //~| expected u64, found u8
    id_u64(b16);
    //~^ ERROR mismatched types
    //~| expected u64, found u16
    id_u64(b32);
    //~^ ERROR mismatched types
    //~| expected u64, found u32
    id_u64(b64); // ok
}
