//@ add-core-stubs
//@ compile-flags: -Cno-prepopulate-passes -Copt-level=0

//@ revisions:x86_64 i686 aarch64-apple aarch64-windows aarch64-linux arm riscv

//@[x86_64] compile-flags: --target x86_64-unknown-uefi
//@[x86_64] needs-llvm-components: x86
//@[i686] compile-flags: --target i686-unknown-linux-musl
//@[i686] needs-llvm-components: x86
//@[aarch64-windows] compile-flags: --target aarch64-pc-windows-msvc
//@[aarch64-windows] needs-llvm-components: aarch64
//@[aarch64-linux] compile-flags: --target aarch64-unknown-linux-gnu
//@[aarch64-linux] needs-llvm-components: aarch64
//@[aarch64-apple] compile-flags: --target aarch64-apple-darwin
//@[aarch64-apple] needs-llvm-components: aarch64
//@[arm] compile-flags: --target armv7r-none-eabi
//@[arm] needs-llvm-components: arm
//@[riscv] compile-flags: --target riscv64gc-unknown-none-elf
//@[riscv] needs-llvm-components: riscv

// See bottom of file for a corresponding C source file that is meant to yield
// equivalent declarations.
#![feature(no_core, lang_items)]
#![crate_type = "lib"]
#![no_std]
#![no_core]

extern crate minicore;
use minicore::*;

// The patterns in this file are written in the style of a table to make the
// uniformities and distinctions more apparent.
//
//                  ZERO/SIGN-EXTENDING TO 32 BITS            NON-EXTENDING
//                  ==============================  =======================
// x86_64:          void @c_arg_u8(i8 zeroext %_a)
// i686:            void @c_arg_u8(i8 zeroext %_a)
// aarch64-apple:   void @c_arg_u8(i8 zeroext %_a)
// aarch64-windows:                                  void @c_arg_u8(i8 %_a)
// aarch64-linux:                                    void @c_arg_u8(i8 %_a)
// arm:             void @c_arg_u8(i8 zeroext %_a)
// riscv:           void @c_arg_u8(i8 zeroext %_a)
#[no_mangle]
pub extern "C" fn c_arg_u8(_a: u8) {}

// x86_64:          void @c_arg_u16(i16 zeroext %_a)
// i686:            void @c_arg_u16(i16 zeroext %_a)
// aarch64-apple:   void @c_arg_u16(i16 zeroext %_a)
// aarch64-windows:                                 void @c_arg_u16(i16 %_a)
// aarch64-linux:                                   void @c_arg_u16(i16 %_a)
// arm:             void @c_arg_u16(i16 zeroext %_a)
// riscv:           void @c_arg_u16(i16 zeroext %_a)
#[no_mangle]
pub extern "C" fn c_arg_u16(_a: u16) {}

// x86_64:          void @c_arg_u32(i32 %_a)
// i686:            void @c_arg_u32(i32 %_a)
// aarch64-apple:   void @c_arg_u32(i32 %_a)
// aarch64-windows:                                 void @c_arg_u32(i32 %_a)
// aarch64-linux:                                   void @c_arg_u32(i32 %_a)
// arm:             void @c_arg_u32(i32 %_a)
// riscv:           void @c_arg_u32(i32 signext %_a)
#[no_mangle]
pub extern "C" fn c_arg_u32(_a: u32) {}

// x86_64:          void @c_arg_u64(i64 %_a)
// i686:            void @c_arg_u64(i64 %_a)
// aarch64-apple:   void @c_arg_u64(i64 %_a)
// aarch64-windows:                                 void @c_arg_u64(i64 %_a)
// aarch64-linux:                                   void @c_arg_u64(i64 %_a)
// arm:             void @c_arg_u64(i64 %_a)
// riscv:           void @c_arg_u64(i64 %_a)
#[no_mangle]
pub extern "C" fn c_arg_u64(_a: u64) {}

// x86_64:          void @c_arg_i8(i8 signext %_a)
// i686:            void @c_arg_i8(i8 signext %_a)
// aarch64-apple:   void @c_arg_i8(i8 signext %_a)
// aarch64-windows:                                  void @c_arg_i8(i8 %_a)
// aarch64-linux:                                    void @c_arg_i8(i8 %_a)
// arm:             void @c_arg_i8(i8 signext %_a)
// riscv:           void @c_arg_i8(i8 signext %_a)
#[no_mangle]
pub extern "C" fn c_arg_i8(_a: i8) {}

// x86_64:          void @c_arg_i16(i16 signext %_a)
// i686:            void @c_arg_i16(i16 signext %_a)
// aarch64-apple:   void @c_arg_i16(i16 signext %_a)
// aarch64-windows:                                 void @c_arg_i16(i16 %_a)
// aarch64-linux:                                   void @c_arg_i16(i16 %_a)
// arm:             void @c_arg_i16(i16 signext %_a)
// riscv:           void @c_arg_i16(i16 signext %_a)
#[no_mangle]
pub extern "C" fn c_arg_i16(_a: i16) {}

// x86_64:          void @c_arg_i32(i32 %_a)
// i686:            void @c_arg_i32(i32 %_a)
// aarch64-apple:   void @c_arg_i32(i32 %_a)
// aarch64-windows:                                 void @c_arg_i32(i32 %_a)
// aarch64-linux:                                   void @c_arg_i32(i32 %_a)
// arm:             void @c_arg_i32(i32 %_a)
// riscv:           void @c_arg_i32(i32 signext %_a)
#[no_mangle]
pub extern "C" fn c_arg_i32(_a: i32) {}

// x86_64:          void @c_arg_i64(i64 %_a)
// i686:            void @c_arg_i64(i64 %_a)
// aarch64-apple:   void @c_arg_i64(i64 %_a)
// aarch64-windows:                                 void @c_arg_i64(i64 %_a)
// aarch64-linux:                                   void @c_arg_i64(i64 %_a)
// arm:             void @c_arg_i64(i64 %_a)
// riscv:           void @c_arg_i64(i64 %_a)
#[no_mangle]
pub extern "C" fn c_arg_i64(_a: i64) {}

// x86_64:          zeroext i8 @c_ret_u8()
// i686:            zeroext i8 @c_ret_u8()
// aarch64-apple:   zeroext i8 @c_ret_u8()
// aarch64-windows:                                 i8 @c_ret_u8()
// aarch64-linux:                                   i8 @c_ret_u8()
// arm:             zeroext i8 @c_ret_u8()
// riscv:           zeroext i8 @c_ret_u8()
#[no_mangle]
pub extern "C" fn c_ret_u8() -> u8 {
    0
}

// x86_64:          zeroext i16 @c_ret_u16()
// i686:            zeroext i16 @c_ret_u16()
// aarch64-apple:   zeroext i16 @c_ret_u16()
// aarch64-windows:                                 i16 @c_ret_u16()
// aarch64-linux:                                   i16 @c_ret_u16()
// arm:             zeroext i16 @c_ret_u16()
// riscv:           zeroext i16 @c_ret_u16()
#[no_mangle]
pub extern "C" fn c_ret_u16() -> u16 {
    0
}

// x86_64:          i32 @c_ret_u32()
// i686:            i32 @c_ret_u32()
// aarch64-apple:   i32 @c_ret_u32()
// aarch64-windows:                                 i32 @c_ret_u32()
// aarch64-linux:                                   i32 @c_ret_u32()
// arm:             i32 @c_ret_u32()
// riscv:           signext i32 @c_ret_u32()
#[no_mangle]
pub extern "C" fn c_ret_u32() -> u32 {
    0
}

// x86_64:          i64 @c_ret_u64()
// i686:            i64 @c_ret_u64()
// aarch64-apple:   i64 @c_ret_u64()
// aarch64-windows:                                 i64 @c_ret_u64()
// aarch64-linux:                                   i64 @c_ret_u64()
// arm:             i64 @c_ret_u64()
// riscv:           i64 @c_ret_u64()
#[no_mangle]
pub extern "C" fn c_ret_u64() -> u64 {
    0
}

// x86_64:          signext i8 @c_ret_i8()
// i686:            signext i8 @c_ret_i8()
// aarch64-apple:   signext i8 @c_ret_i8()
// aarch64-windows:                                 i8 @c_ret_i8()
// aarch64-linux:                                   i8 @c_ret_i8()
// arm:             signext i8 @c_ret_i8()
// riscv:           signext i8 @c_ret_i8()
#[no_mangle]
pub extern "C" fn c_ret_i8() -> i8 {
    0
}

// x86_64:          signext i16 @c_ret_i16()
// i686:            signext i16 @c_ret_i16()
// aarch64-apple:   signext i16 @c_ret_i16()
// aarch64-windows:                                 i16 @c_ret_i16()
// aarch64-linux:                                   i16 @c_ret_i16()
// arm:             signext i16 @c_ret_i16()
// riscv:           signext i16 @c_ret_i16()
#[no_mangle]
pub extern "C" fn c_ret_i16() -> i16 {
    0
}

// x86_64:          i32 @c_ret_i32()
// i686:            i32 @c_ret_i32()
// aarch64-apple:   i32 @c_ret_i32()
// aarch64-windows:                                 i32 @c_ret_i32()
// aarch64-linux:                                   i32 @c_ret_i32()
// arm:             i32 @c_ret_i32()
// riscv:           signext i32 @c_ret_i32()
#[no_mangle]
pub extern "C" fn c_ret_i32() -> i32 {
    0
}

// x86_64:          i64 @c_ret_i64()
// i686:            i64 @c_ret_i64()
// aarch64-apple:   i64 @c_ret_i64()
// aarch64-windows:                                 i64 @c_ret_i64()
// aarch64-linux:                                   i64 @c_ret_i64()
// arm:             i64 @c_ret_i64()
// riscv:           i64 @c_ret_i64()
#[no_mangle]
pub extern "C" fn c_ret_i64() -> i64 {
    0
}

const C_SOURCE_FILE: &'static str = r##"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

void c_arg_u8(uint8_t _a) { }
void c_arg_u16(uint16_t _a) { }
void c_arg_u32(uint32_t _a) { }
void c_arg_u64(uint64_t _a) { }

void c_arg_i8(int8_t _a) { }
void c_arg_i16(int16_t _a) { }
void c_arg_i32(int32_t _a) { }
void c_arg_i64(int64_t _a) { }

uint8_t  c_ret_u8()  { return 0; }
uint16_t c_ret_u16() { return 0; }
uint32_t c_ret_u32() { return 0; }
uint64_t c_ret_u64() { return 0; }

int8_t   c_ret_i8()  { return 0; }
int16_t  c_ret_i16() { return 0; }
int32_t  c_ret_i32() { return 0; }
int64_t  c_ret_i64() { return 0; }
"##;
