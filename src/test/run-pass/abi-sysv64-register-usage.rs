// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Checks if the correct registers are being used to pass arguments
// when the sysv64 ABI is specified.

// ignore-android
// ignore-arm
// ignore-aarch64

#![feature(abi_sysv64)]
#![feature(asm)]

#[cfg(target_arch = "x86_64")]
pub extern "sysv64" fn all_the_registers(rdi: i64, rsi: i64, rdx: i64,
                                         rcx: i64, r8 : i64, r9 : i64,
                                         xmm0: f32, xmm1: f32, xmm2: f32,
                                         xmm3: f32, xmm4: f32, xmm5: f32,
                                         xmm6: f32, xmm7: f32) -> i64 {
    assert_eq!(rdi, 1);
    assert_eq!(rsi, 2);
    assert_eq!(rdx, 3);
    assert_eq!(rcx, 4);
    assert_eq!(r8,  5);
    assert_eq!(r9,  6);
    assert_eq!(xmm0, 1.0f32);
    assert_eq!(xmm1, 2.0f32);
    assert_eq!(xmm2, 4.0f32);
    assert_eq!(xmm3, 8.0f32);
    assert_eq!(xmm4, 16.0f32);
    assert_eq!(xmm5, 32.0f32);
    assert_eq!(xmm6, 64.0f32);
    assert_eq!(xmm7, 128.0f32);
    42
}

// this struct contains 8 i64's, while only 6 can be passed in registers.
#[cfg(target_arch = "x86_64")]
#[derive(PartialEq, Eq, Debug)]
pub struct LargeStruct(i64, i64, i64, i64, i64, i64, i64, i64);

#[cfg(target_arch = "x86_64")]
#[inline(never)]
pub extern "sysv64" fn large_struct_by_val(mut foo: LargeStruct) -> LargeStruct {
    foo.0 *= 1;
    foo.1 *= 2;
    foo.2 *= 3;
    foo.3 *= 4;
    foo.4 *= 5;
    foo.5 *= 6;
    foo.6 *= 7;
    foo.7 *= 8;
    foo
}

#[cfg(target_arch = "x86_64")]
pub fn main() {
    let result: i64;
    unsafe {
        asm!("mov rdi, 1;
              mov rsi, 2;
              mov rdx, 3;
              mov rcx, 4;
              mov r8,  5;
              mov r9,  6;
              mov eax, 0x3F800000;
              movd xmm0, eax;
              mov eax, 0x40000000;
              movd xmm1, eax;
              mov eax, 0x40800000;
              movd xmm2, eax;
              mov eax, 0x41000000;
              movd xmm3, eax;
              mov eax, 0x41800000;
              movd xmm4, eax;
              mov eax, 0x42000000;
              movd xmm5, eax;
              mov eax, 0x42800000;
              movd xmm6, eax;
              mov eax, 0x43000000;
              movd xmm7, eax;
              call r10
              "
            : "={rax}"(result)
            : "{r10}"(all_the_registers as usize)
            : "rdi", "rsi", "rdx", "rcx", "r8", "r9", "r11", "cc", "memory"
            : "intel", "alignstack"
        )
    }
    assert_eq!(result, 42);

    assert_eq!(
        large_struct_by_val(LargeStruct(1, 2, 3, 4, 5, 6, 7, 8)),
        LargeStruct(1, 4, 9, 16, 25, 36, 49, 64)
    );
}

#[cfg(not(target_arch = "x86_64"))]
pub fn main() {}