// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test a foreign function that accepts empty struct.

// pretty-expanded FIXME #23616
// ignore-msvc
// ignore-emscripten

#[repr(C)]
struct TwoU8s {
    one: u8,
    two: u8,
}

#[repr(C)]
struct ManyInts {
    arg1: i8,
    arg2: i16,
    arg3: i32,
    arg4: i16,
    arg5: i8,
    arg6: TwoU8s,
}

#[repr(C)]
struct Empty;

#[link(name = "rust_test_helpers", kind = "static")]
extern {
    fn rust_dbg_extern_empty_struct(v1: ManyInts, e: Empty, v2: ManyInts);
}

pub fn main() {
    unsafe {
        let x = ManyInts {
            arg1: 2,
            arg2: 3,
            arg3: 4,
            arg4: 5,
            arg5: 6,
            arg6: TwoU8s { one: 7, two: 8, }
        };
        let y = ManyInts {
            arg1: 1,
            arg2: 2,
            arg3: 3,
            arg4: 4,
            arg5: 5,
            arg6: TwoU8s { one: 6, two: 7, }
        };
        let empty = Empty;
        rust_dbg_extern_empty_struct(x, empty, y);
    }
}
