// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-tidy-linelength

// Issue 23030: Detect overflowing discriminant
//
// Check that we detect the overflow even if enum is not used.

// See also run-pass/discrim-explicit-23030.rs where the suggested
// workaround is tested.

use std::{i8,u8,i16,u16,i32,u32,i64, u64};

fn f_i8() {
    #[repr(i8)]
    enum A {
        Ok = i8::MAX - 1,
        Ok2,
        OhNo, //~ ERROR enum discriminant overflowed [E0370]
              //~| NOTE overflowed on value after 127i8
              //~| NOTE explicitly set `OhNo = -128i8` if that is desired outcome
    }
}

fn f_u8() {
    #[repr(u8)]
    enum A {
        Ok = u8::MAX - 1,
        Ok2,
        OhNo, //~ ERROR enum discriminant overflowed [E0370]
              //~| NOTE overflowed on value after 255u8
              //~| NOTE explicitly set `OhNo = 0u8` if that is desired outcome
    }
}

fn f_i16() {
    #[repr(i16)]
    enum A {
        Ok = i16::MAX - 1,
        Ok2,
        OhNo, //~ ERROR enum discriminant overflowed [E0370]
              //~| NOTE overflowed on value after 32767i16
              //~| NOTE explicitly set `OhNo = -32768i16` if that is desired outcome
    }
}

fn f_u16() {
    #[repr(u16)]
    enum A {
        Ok = u16::MAX - 1,
        Ok2,
        OhNo, //~ ERROR enum discriminant overflowed [E0370]
              //~| NOTE overflowed on value after 65535u16
              //~| NOTE explicitly set `OhNo = 0u16` if that is desired outcome
    }
}

fn f_i32() {
    #[repr(i32)]
    enum A {
        Ok = i32::MAX - 1,
        Ok2,
        OhNo, //~ ERROR enum discriminant overflowed [E0370]
              //~| NOTE overflowed on value after 2147483647i32
              //~| NOTE explicitly set `OhNo = -2147483648i32` if that is desired outcome
    }
}

fn f_u32() {
    #[repr(u32)]
    enum A {
        Ok = u32::MAX - 1,
        Ok2,
        OhNo, //~ ERROR enum discriminant overflowed [E0370]
              //~| NOTE overflowed on value after 4294967295u32
              //~| NOTE explicitly set `OhNo = 0u32` if that is desired outcome
    }
}

fn f_i64() {
    #[repr(i64)]
    enum A {
        Ok = i64::MAX - 1,
        Ok2,
        OhNo, //~ ERROR enum discriminant overflowed [E0370]
              //~| NOTE overflowed on value after 9223372036854775807i64
              //~| NOTE explicitly set `OhNo = -9223372036854775808i64` if that is desired outcome
    }
}

fn f_u64() {
    #[repr(u64)]
    enum A {
        Ok = u64::MAX - 1,
        Ok2,
        OhNo, //~ ERROR enum discriminant overflowed [E0370]
              //~| NOTE overflowed on value after 18446744073709551615u64
              //~| NOTE explicitly set `OhNo = 0u64` if that is desired outcome
    }
}

fn main() { }
