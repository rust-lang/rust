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

#[lang = "sized"]
trait Sized {}
#[lang = "freeze"]
trait Freeze {}
#[lang = "copy"]
trait Copy {}

// The patterns in this file are written in the style of a table to make the
// uniformities and distinctions more apparent.
//
//                  ZERO/SIGN-EXTENDING TO 32 BITS            NON-EXTENDING
//                  ==============================  =======================
// CHECK-X86_64:          void @c_arg_u8(i8 zeroext %_a)
// CHECK-I686:            void @c_arg_u8(i8 zeroext %_a)
// CHECK-AARCH64-APPLE:   void @c_arg_u8(i8 zeroext %_a)
// CHECK-AARCH64-WINDOWS:                                  void @c_arg_u8(i8 %_a)
// CHECK-AARCH64-LINUX:                                    void @c_arg_u8(i8 %_a)
// CHECK-ARM:             void @c_arg_u8(i8 zeroext %_a)
// CHECK-RISCV:           void @c_arg_u8(i8 zeroext %_a)
#[no_mangle]
pub extern "C" fn c_arg_u8(_a: u8) {}

// CHECK-X86_64:          void @c_arg_u16(i16 zeroext %_a)
// CHECK-I686:            void @c_arg_u16(i16 zeroext %_a)
// CHECK-AARCH64-APPLE:   void @c_arg_u16(i16 zeroext %_a)
// CHECK-AARCH64-WINDOWS:                                 void @c_arg_u16(i16 %_a)
// CHECK-AARCH64-LINUX:                                   void @c_arg_u16(i16 %_a)
// CHECK-ARM:             void @c_arg_u16(i16 zeroext %_a)
// CHECK-RISCV:           void @c_arg_u16(i16 zeroext %_a)
#[no_mangle]
pub extern "C" fn c_arg_u16(_a: u16) {}

// CHECK-X86_64:          void @c_arg_u32(i32 %_a)
// CHECK-I686:            void @c_arg_u32(i32 %_a)
// CHECK-AARCH64-APPLE:   void @c_arg_u32(i32 %_a)
// CHECK-AARCH64-WINDOWS:                                 void @c_arg_u32(i32 %_a)
// CHECK-AARCH64-LINUX:                                   void @c_arg_u32(i32 %_a)
// CHECK-ARM:             void @c_arg_u32(i32 %_a)
// CHECK-RISCV:           void @c_arg_u32(i32 signext %_a)
#[no_mangle]
pub extern "C" fn c_arg_u32(_a: u32) {}

// CHECK-X86_64:          void @c_arg_u64(i64 %_a)
// CHECK-I686:            void @c_arg_u64(i64 %_a)
// CHECK-AARCH64-APPLE:   void @c_arg_u64(i64 %_a)
// CHECK-AARCH64-WINDOWS:                                 void @c_arg_u64(i64 %_a)
// CHECK-AARCH64-LINUX:                                   void @c_arg_u64(i64 %_a)
// CHECK-ARM:             void @c_arg_u64(i64 %_a)
// CHECK-RISCV:           void @c_arg_u64(i64 %_a)
#[no_mangle]
pub extern "C" fn c_arg_u64(_a: u64) {}

// CHECK-X86_64:          void @c_arg_i8(i8 signext %_a)
// CHECK-I686:            void @c_arg_i8(i8 signext %_a)
// CHECK-AARCH64-APPLE:   void @c_arg_i8(i8 signext %_a)
// CHECK-AARCH64-WINDOWS:                                  void @c_arg_i8(i8 %_a)
// CHECK-AARCH64-LINUX:                                    void @c_arg_i8(i8 %_a)
// CHECK-ARM:             void @c_arg_i8(i8 signext %_a)
// CHECK-RISCV:           void @c_arg_i8(i8 signext %_a)
#[no_mangle]
pub extern "C" fn c_arg_i8(_a: i8) {}

// CHECK-X86_64:          void @c_arg_i16(i16 signext %_a)
// CHECK-I686:            void @c_arg_i16(i16 signext %_a)
// CHECK-AARCH64-APPLE:   void @c_arg_i16(i16 signext %_a)
// CHECK-AARCH64-WINDOWS:                                 void @c_arg_i16(i16 %_a)
// CHECK-AARCH64-LINUX:                                   void @c_arg_i16(i16 %_a)
// CHECK-ARM:             void @c_arg_i16(i16 signext %_a)
// CHECK-RISCV:           void @c_arg_i16(i16 signext %_a)
#[no_mangle]
pub extern "C" fn c_arg_i16(_a: i16) {}

// CHECK-X86_64:          void @c_arg_i32(i32 %_a)
// CHECK-I686:            void @c_arg_i32(i32 %_a)
// CHECK-AARCH64-APPLE:   void @c_arg_i32(i32 %_a)
// CHECK-AARCH64-WINDOWS:                                 void @c_arg_i32(i32 %_a)
// CHECK-AARCH64-LINUX:                                   void @c_arg_i32(i32 %_a)
// CHECK-ARM:             void @c_arg_i32(i32 %_a)
// CHECK-RISCV:           void @c_arg_i32(i32 signext %_a)
#[no_mangle]
pub extern "C" fn c_arg_i32(_a: i32) {}

// CHECK-X86_64:          void @c_arg_i64(i64 %_a)
// CHECK-I686:            void @c_arg_i64(i64 %_a)
// CHECK-AARCH64-APPLE:   void @c_arg_i64(i64 %_a)
// CHECK-AARCH64-WINDOWS:                                 void @c_arg_i64(i64 %_a)
// CHECK-AARCH64-LINUX:                                   void @c_arg_i64(i64 %_a)
// CHECK-ARM:             void @c_arg_i64(i64 %_a)
// CHECK-RISCV:           void @c_arg_i64(i64 %_a)
#[no_mangle]
pub extern "C" fn c_arg_i64(_a: i64) {}

// CHECK-X86_64:          zeroext i8 @c_ret_u8()
// CHECK-I686:            zeroext i8 @c_ret_u8()
// CHECK-AARCH64-APPLE:   zeroext i8 @c_ret_u8()
// CHECK-AARCH64-WINDOWS:                                 i8 @c_ret_u8()
// CHECK-AARCH64-LINUX:                                   i8 @c_ret_u8()
// CHECK-ARM:             zeroext i8 @c_ret_u8()
// CHECK-RISCV:           zeroext i8 @c_ret_u8()
#[no_mangle]
pub extern "C" fn c_ret_u8() -> u8 {
    0
}

// CHECK-X86_64:          zeroext i16 @c_ret_u16()
// CHECK-I686:            zeroext i16 @c_ret_u16()
// CHECK-AARCH64-APPLE:   zeroext i16 @c_ret_u16()
// CHECK-AARCH64-WINDOWS:                                 i16 @c_ret_u16()
// CHECK-AARCH64-LINUX:                                   i16 @c_ret_u16()
// CHECK-ARM:             zeroext i16 @c_ret_u16()
// CHECK-RISCV:           zeroext i16 @c_ret_u16()
#[no_mangle]
pub extern "C" fn c_ret_u16() -> u16 {
    0
}

// CHECK-X86_64:          i32 @c_ret_u32()
// CHECK-I686:            i32 @c_ret_u32()
// CHECK-AARCH64-APPLE:   i32 @c_ret_u32()
// CHECK-AARCH64-WINDOWS:                                 i32 @c_ret_u32()
// CHECK-AARCH64-LINUX:                                   i32 @c_ret_u32()
// CHECK-ARM:             i32 @c_ret_u32()
// CHECK-RISCV:           signext i32 @c_ret_u32()
#[no_mangle]
pub extern "C" fn c_ret_u32() -> u32 {
    0
}

// CHECK-X86_64:          i64 @c_ret_u64()
// CHECK-I686:            i64 @c_ret_u64()
// CHECK-AARCH64-APPLE:   i64 @c_ret_u64()
// CHECK-AARCH64-WINDOWS:                                 i64 @c_ret_u64()
// CHECK-AARCH64-LINUX:                                   i64 @c_ret_u64()
// CHECK-ARM:             i64 @c_ret_u64()
// CHECK-RISCV:           i64 @c_ret_u64()
#[no_mangle]
pub extern "C" fn c_ret_u64() -> u64 {
    0
}

// CHECK-X86_64:          signext i8 @c_ret_i8()
// CHECK-I686:            signext i8 @c_ret_i8()
// CHECK-AARCH64-APPLE:   signext i8 @c_ret_i8()
// CHECK-AARCH64-WINDOWS:                                 i8 @c_ret_i8()
// CHECK-AARCH64-LINUX:                                   i8 @c_ret_i8()
// CHECK-ARM:             signext i8 @c_ret_i8()
// CHECK-RISCV:           signext i8 @c_ret_i8()
#[no_mangle]
pub extern "C" fn c_ret_i8() -> i8 {
    0
}

// CHECK-X86_64:          signext i16 @c_ret_i16()
// CHECK-I686:            signext i16 @c_ret_i16()
// CHECK-AARCH64-APPLE:   signext i16 @c_ret_i16()
// CHECK-AARCH64-WINDOWS:                                 i16 @c_ret_i16()
// CHECK-AARCH64-LINUX:                                   i16 @c_ret_i16()
// CHECK-ARM:             signext i16 @c_ret_i16()
// CHECK-RISCV:           signext i16 @c_ret_i16()
#[no_mangle]
pub extern "C" fn c_ret_i16() -> i16 {
    0
}

// CHECK-X86_64:          i32 @c_ret_i32()
// CHECK-I686:            i32 @c_ret_i32()
// CHECK-AARCH64-APPLE:   i32 @c_ret_i32()
// CHECK-AARCH64-WINDOWS:                                 i32 @c_ret_i32()
// CHECK-AARCH64-LINUX:                                   i32 @c_ret_i32()
// CHECK-ARM:             i32 @c_ret_i32()
// CHECK-RISCV:           signext i32 @c_ret_i32()
#[no_mangle]
pub extern "C" fn c_ret_i32() -> i32 {
    0
}

// CHECK-X86_64:          i64 @c_ret_i64()
// CHECK-I686:            i64 @c_ret_i64()
// CHECK-AARCH64-APPLE:   i64 @c_ret_i64()
// CHECK-AARCH64-WINDOWS:                                 i64 @c_ret_i64()
// CHECK-AARCH64-LINUX:                                   i64 @c_ret_i64()
// CHECK-ARM:             i64 @c_ret_i64()
// CHECK-RISCV:           i64 @c_ret_i64()
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
