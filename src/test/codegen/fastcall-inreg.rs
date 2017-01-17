// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Checks if the "fastcall" calling convention marks function arguments
// as "inreg" like the C/C++ compilers for the platforms.
// x86 only.

// ignore-aarch64
// ignore-aarch64_be
// ignore-arm
// ignore-armeb
// ignore-avr
// ignore-bpfel
// ignore-bpfeb
// ignore-hexagon
// ignore-mips
// ignore-mipsel
// ignore-mips64
// ignore-mips64el
// ignore-msp430
// ignore-powerpc64
// ignore-powerpc64le
// ignore-powerpc
// ignore-r600
// ignore-amdgcn
// ignore-sparc
// ignore-sparcv9
// ignore-sparcel
// ignore-s390x
// ignore-tce
// ignore-thumb
// ignore-thumbeb
// ignore-x86_64 no-ignore-x86
// ignore-xcore
// ignore-nvptx
// ignore-nvptx64
// ignore-le32
// ignore-le64
// ignore-amdil
// ignore-amdil64
// ignore-hsail
// ignore-hsail64
// ignore-spir
// ignore-spir64
// ignore-kalimba
// ignore-shave
// ignore-wasm32
// ignore-wasm64
// ignore-emscripten

// compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]

mod tests {
    // CHECK: @f1(i32 inreg, i32 inreg, i32)
    #[no_mangle]
    extern "fastcall" fn f1(_: i32, _: i32, _: i32) {}

    // CHECK: @f2(i32* inreg, i32* inreg, i32*)
    #[no_mangle]
    extern "fastcall" fn f2(_: *const i32, _: *const i32, _: *const i32) {}

    // CHECK: @f3(float, i32 inreg, i32 inreg, i32)
    #[no_mangle]
    extern "fastcall" fn f3(_: f32, _: i32, _: i32, _: i32) {}

    // CHECK: @f4(i32 inreg, float, i32 inreg, i32)
    #[no_mangle]
    extern "fastcall" fn f4(_: i32, _: f32, _: i32, _: i32) {}

    // CHECK: @f5(i64, i32)
    #[no_mangle]
    extern "fastcall" fn f5(_: i64, _: i32) {}

    // CHECK: @f6(i1 inreg zeroext, i32 inreg, i32)
    #[no_mangle]
    extern "fastcall" fn f6(_: bool, _: i32, _: i32) {}
}
