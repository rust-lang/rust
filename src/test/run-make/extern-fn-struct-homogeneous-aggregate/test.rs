// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test calls to c functions with various lengths of homogeneous
// floating point aggregates. Lengths are chosen to hit corner cases on
// the ppc64 and aarch64 parameter passing ABIs.

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
struct TwoFloats {
    a: f32,
    b: f32
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
struct FourFloats {
    a: f32,
    b: f32,
    c: f32,
    d: f32
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
struct EightFloats {
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    e: f32,
    f: f32,
    g: f32,
    h: f32
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
struct NineFloats {
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    e: f32,
    f: f32,
    g: f32,
    h: f32,
    i: f32
}

#[link(name = "test", kind = "static")]
extern {
    fn test_two_floats(a: TwoFloats) -> TwoFloats;
    fn test_four_floats(a: FourFloats) -> FourFloats;
    fn test_eight_floats(a: EightFloats) -> EightFloats;
    fn test_nine_floats(a: NineFloats) -> NineFloats;
}

fn main() {
    let a = TwoFloats { a: 2001.0, b: 2002.0 };
    let b = FourFloats { a: 4001.0, b: 4002.0, c: 4003.0, d: 4004.0 };
    let c = EightFloats { a: 8001.0, b: 8002.0, c: 8003.0, d: 8004.0,
                          e: 8005.0, f: 8006.0, g: 8007.0, h: 8008.0 };
    let d = NineFloats { a: 9001.0, b: 9002.0, c: 9003.0, d: 9004.0, e: 9005.0,
                         f: 9006.0, g: 9007.0, h: 9008.0, i: 9009.0 };

    unsafe {
        assert_eq!(test_two_floats(a), a);
        assert_eq!(test_four_floats(b), b);
        assert_eq!(test_eight_floats(c), c);
        assert_eq!(test_nine_floats(d), d);
    }
}
