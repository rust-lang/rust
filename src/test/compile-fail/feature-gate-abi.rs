// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// gate-test-intrinsics
// gate-test-platform_intrinsics
// gate-test-abi_vectorcall

// Functions
extern "rust-intrinsic" fn f1() {} //~ ERROR intrinsics are subject to change
extern "platform-intrinsic" fn f2() {} //~ ERROR platform intrinsics are experimental
extern "vectorcall" fn f3() {} //~ ERROR vectorcall is experimental and subject to change
extern "rust-call" fn f4() {} //~ ERROR rust-call ABI is subject to change
extern "msp430-interrupt" fn f5() {} //~ ERROR msp430-interrupt ABI is experimental

// Methods in trait definition
trait Tr {
    extern "rust-intrinsic" fn m1(); //~ ERROR intrinsics are subject to change
    extern "platform-intrinsic" fn m2(); //~ ERROR platform intrinsics are experimental
    extern "vectorcall" fn m3(); //~ ERROR vectorcall is experimental and subject to change
    extern "rust-call" fn m4(); //~ ERROR rust-call ABI is subject to change
    extern "msp430-interrupt" fn m5(); //~ ERROR msp430-interrupt ABI is experimental

    extern "rust-intrinsic" fn dm1() {} //~ ERROR intrinsics are subject to change
    extern "platform-intrinsic" fn dm2() {} //~ ERROR platform intrinsics are experimental
    extern "vectorcall" fn dm3() {} //~ ERROR vectorcall is experimental and subject to change
    extern "rust-call" fn dm4() {} //~ ERROR rust-call ABI is subject to change
    extern "msp430-interrupt" fn dm5() {} //~ ERROR msp430-interrupt ABI is experimental
}

struct S;

// Methods in trait impl
impl Tr for S {
    extern "rust-intrinsic" fn m1() {} //~ ERROR intrinsics are subject to change
    extern "platform-intrinsic" fn m2() {} //~ ERROR platform intrinsics are experimental
    extern "vectorcall" fn m3() {} //~ ERROR vectorcall is experimental and subject to change
    extern "rust-call" fn m4() {} //~ ERROR rust-call ABI is subject to change
    extern "msp430-interrupt" fn m5() {} //~ ERROR msp430-interrupt ABI is experimental
}

// Methods in inherent impl
impl S {
    extern "rust-intrinsic" fn im1() {} //~ ERROR intrinsics are subject to change
    extern "platform-intrinsic" fn im2() {} //~ ERROR platform intrinsics are experimental
    extern "vectorcall" fn im3() {} //~ ERROR vectorcall is experimental and subject to change
    extern "rust-call" fn im4() {} //~ ERROR rust-call ABI is subject to change
    extern "msp430-interrupt" fn im5() {} //~ ERROR msp430-interrupt ABI is experimental
}

// Function pointer types
type A1 = extern "rust-intrinsic" fn(); //~ ERROR intrinsics are subject to change
type A2 = extern "platform-intrinsic" fn(); //~ ERROR platform intrinsics are experimental
type A3 = extern "vectorcall" fn(); //~ ERROR vectorcall is experimental and subject to change
type A4 = extern "rust-call" fn(); //~ ERROR rust-call ABI is subject to change
type A5 = extern "msp430-interrupt" fn(); //~ ERROR msp430-interrupt ABI is experimental

// Foreign modules
extern "rust-intrinsic" {} //~ ERROR intrinsics are subject to change
extern "platform-intrinsic" {} //~ ERROR platform intrinsics are experimental
extern "vectorcall" {} //~ ERROR vectorcall is experimental and subject to change
extern "rust-call" {} //~ ERROR rust-call ABI is subject to change
extern "msp430-interrupt" {} //~ ERROR msp430-interrupt ABI is experimental

fn main() {}
