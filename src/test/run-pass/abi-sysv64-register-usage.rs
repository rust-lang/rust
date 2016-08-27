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

#![feature(abi_sysv64)]
#![feature(naked_functions)]
#![feature(asm)]

#[naked]
#[inline(never)]
#[allow(unused_variables)]
pub unsafe extern "sysv64" fn all_the_registers(rdi: i64, rsi: i64, rdx: i64,
                                                rcx: i64, r8 : i64, r9 : i64,
                                                xmm0: f32, xmm1: f32, xmm2: f32,
                                                xmm3: f32, xmm4: f32, xmm5: f32,
                                                xmm6: f32, xmm7: f32) -> i64 {
    // this assembly checks all registers for specific values, and puts in rax
    // how many values were correct.
    asm!("cmp rdi, 0x1;
          xor rax, rax;
          setz al;

          cmp rsi, 0x2;
          xor rdi, rdi
          setz dil;
          add rax, rdi;

          cmp rdx, 0x3;
          setz dil;
          add rax, rdi;

          cmp rcx, 0x4;
          setz dil;
          add rax, rdi;

          cmp r8, 0x5;
          setz dil;
          add rax, rdi;

          cmp r9, 0x6;
          setz dil;
          add rax, rdi;

          movd esi, xmm0;
          cmp rsi, 0x3F800000;
          setz dil;
          add rax, rdi;

          movd esi, xmm1;
          cmp rsi, 0x40000000;
          setz dil;
          add rax, rdi;

          movd esi, xmm2;
          cmp rsi, 0x40800000;
          setz dil;
          add rax, rdi;

          movd esi, xmm3;
          cmp rsi, 0x41000000;
          setz dil;
          add rax, rdi;

          movd esi, xmm4;
          cmp rsi, 0x41800000;
          setz dil;
          add rax, rdi;

          movd esi, xmm5;
          cmp rsi, 0x42000000;
          setz dil;
          add rax, rdi;

          movd esi, xmm6;
          cmp rsi, 0x42800000;
          setz dil;
          add rax, rdi;

          movd esi, xmm7;
          cmp rsi, 0x43000000;
          setz dil;
          add rax, rdi;
          ret
         " :::: "intel");
    unreachable!();
}

// this struct contains 8 i64's, while only 6 can be passed in registers.
#[derive(PartialEq, Eq, Debug)]
pub struct LargeStruct(i64, i64, i64, i64, i64, i64, i64, i64);

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

pub fn main() {
    assert_eq!(unsafe {
        all_the_registers(1, 2, 3, 4, 5, 6,
                          1.0, 2.0, 4.0, 8.0,
                          16.0, 32.0, 64.0, 128.0)
    }, 14);

    assert_eq!(
        large_struct_by_val(LargeStruct(1, 2, 3, 4, 5, 6, 7, 8)),
        LargeStruct(1, 4, 9, 16, 25, 36, 49, 64)
    );
}
