// Test that there's no conditional trap for zero divisor for all mips
// targets by default.  Division by zero is defined as panic so the trap is
// redundant.
//
//@ add-minicore
//@ assembly-output: emit-asm
//
// See https://github.com/llvm/llvm-project/pull/204386.
//@ compile-flags: -Copt-level=3
//
//@ revisions: mips64el-unknown-linux-gnuabi64
//@[mips64el-unknown-linux-gnuabi64] compile-flags: --target=mips64el-unknown-linux-gnuabi64
//@[mips64el-unknown-linux-gnuabi64] needs-llvm-components: mips
//@[mips64el-unknown-linux-gnuabi64] filecheck-flags: --check-prefix NOTRAP
//@ revisions: mips64el-unknown-linux-muslabi64
//@[mips64el-unknown-linux-muslabi64] compile-flags: --target=mips64el-unknown-linux-muslabi64
//@[mips64el-unknown-linux-muslabi64] needs-llvm-components: mips
//@[mips64el-unknown-linux-muslabi64] filecheck-flags: --check-prefix NOTRAP
//@ revisions: mips64-openwrt-linux-musl
//@[mips64-openwrt-linux-musl] compile-flags: --target=mips64-openwrt-linux-musl
//@[mips64-openwrt-linux-musl] needs-llvm-components: mips
//@[mips64-openwrt-linux-musl] filecheck-flags: --check-prefix NOTRAP
//@ revisions: mips64-unknown-linux-gnuabi64
//@[mips64-unknown-linux-gnuabi64] compile-flags: --target=mips64-unknown-linux-gnuabi64
//@[mips64-unknown-linux-gnuabi64] needs-llvm-components: mips
//@[mips64-unknown-linux-gnuabi64] filecheck-flags: --check-prefix NOTRAP
//@ revisions: mips64-unknown-linux-muslabi64
//@[mips64-unknown-linux-muslabi64] compile-flags: --target=mips64-unknown-linux-muslabi64
//@[mips64-unknown-linux-muslabi64] needs-llvm-components: mips
//@[mips64-unknown-linux-muslabi64] filecheck-flags: --check-prefix NOTRAP
//@ revisions: mipsel-mti-none-elf
//@[mipsel-mti-none-elf] compile-flags: --target=mipsel-mti-none-elf
//@[mipsel-mti-none-elf] needs-llvm-components: mips
//@[mipsel-mti-none-elf] filecheck-flags: --check-prefix NOTRAP
//@ revisions: mipsel-sony-psp
//@[mipsel-sony-psp] compile-flags: --target=mipsel-sony-psp
//@[mipsel-sony-psp] needs-llvm-components: mips
//@[mipsel-sony-psp] filecheck-flags: --check-prefix NOTRAP
//@ revisions: mipsel-sony-psx
//@[mipsel-sony-psx] compile-flags: --target=mipsel-sony-psx
//@[mipsel-sony-psx] needs-llvm-components: mips
//@[mipsel-sony-psx] filecheck-flags: --check-prefix NOTRAP
//@ revisions: mipsel-unknown-linux-gnu
//@[mipsel-unknown-linux-gnu] compile-flags: --target=mipsel-unknown-linux-gnu
//@[mipsel-unknown-linux-gnu] needs-llvm-components: mips
//@[mipsel-unknown-linux-gnu] filecheck-flags: --check-prefix NOTRAP
//@ revisions: mipsel-unknown-linux-musl
//@[mipsel-unknown-linux-musl] compile-flags: --target=mipsel-unknown-linux-musl
//@[mipsel-unknown-linux-musl] needs-llvm-components: mips
//@[mipsel-unknown-linux-musl] filecheck-flags: --check-prefix NOTRAP
//@ revisions: mipsel-unknown-linux-uclibc
//@[mipsel-unknown-linux-uclibc] compile-flags: --target=mipsel-unknown-linux-uclibc
//@[mipsel-unknown-linux-uclibc] needs-llvm-components: mips
//@[mipsel-unknown-linux-uclibc] filecheck-flags: --check-prefix NOTRAP
//@ revisions: mipsel-unknown-netbsd
//@[mipsel-unknown-netbsd] compile-flags: --target=mipsel-unknown-netbsd
//@[mipsel-unknown-netbsd] needs-llvm-components: mips
//@[mipsel-unknown-netbsd] filecheck-flags: --check-prefix NOTRAP
//@ revisions: mipsel-unknown-none
//@[mipsel-unknown-none] compile-flags: --target=mipsel-unknown-none
//@[mipsel-unknown-none] needs-llvm-components: mips
//@[mipsel-unknown-none] filecheck-flags: --check-prefix NOTRAP
//@ revisions: mipsisa32r6el-unknown-linux-gnu
//@[mipsisa32r6el-unknown-linux-gnu] compile-flags: --target=mipsisa32r6el-unknown-linux-gnu
//@[mipsisa32r6el-unknown-linux-gnu] needs-llvm-components: mips
//@[mipsisa32r6el-unknown-linux-gnu] filecheck-flags: --check-prefix NOTRAP
//@ revisions: mipsisa32r6-unknown-linux-gnu
//@[mipsisa32r6-unknown-linux-gnu] compile-flags: --target=mipsisa32r6-unknown-linux-gnu
//@[mipsisa32r6-unknown-linux-gnu] needs-llvm-components: mips
//@[mipsisa32r6-unknown-linux-gnu] filecheck-flags: --check-prefix NOTRAP
//@ revisions: mipsisa64r6el-unknown-linux-gnuabi64
//@[mipsisa64r6el-unknown-linux-gnuabi64] compile-flags: --target=mipsisa64r6el-unknown-linux-gnuabi64
//@[mipsisa64r6el-unknown-linux-gnuabi64] needs-llvm-components: mips
//@[mipsisa64r6el-unknown-linux-gnuabi64] filecheck-flags: --check-prefix NOTRAP
//@ revisions: mipsisa64r6-unknown-linux-gnuabi64
//@[mipsisa64r6-unknown-linux-gnuabi64] compile-flags: --target=mipsisa64r6-unknown-linux-gnuabi64
//@[mipsisa64r6-unknown-linux-gnuabi64] needs-llvm-components: mips
//@[mipsisa64r6-unknown-linux-gnuabi64] filecheck-flags: --check-prefix NOTRAP
//@ revisions: mips-mti-none-elf
//@[mips-mti-none-elf] compile-flags: --target=mips-mti-none-elf
//@[mips-mti-none-elf] needs-llvm-components: mips
//@[mips-mti-none-elf] filecheck-flags: --check-prefix NOTRAP
//@ revisions: mips-unknown-linux-gnu
//@[mips-unknown-linux-gnu] compile-flags: --target=mips-unknown-linux-gnu
//@[mips-unknown-linux-gnu] needs-llvm-components: mips
//@[mips-unknown-linux-gnu] filecheck-flags: --check-prefix NOTRAP
//@ revisions: mips-unknown-linux-musl
//@[mips-unknown-linux-musl] compile-flags: --target=mips-unknown-linux-musl
//@[mips-unknown-linux-musl] needs-llvm-components: mips
//@[mips-unknown-linux-musl] filecheck-flags: --check-prefix NOTRAP
//@ revisions: mips-unknown-linux-uclibc
//@[mips-unknown-linux-uclibc] compile-flags: --target=mips-unknown-linux-uclibc
//@[mips-unknown-linux-uclibc] needs-llvm-components: mips
//@[mips-unknown-linux-uclibc] filecheck-flags: --check-prefix NOTRAP
//
//@ revisions: TRAP
//@[TRAP] compile-flags: --target=mips64el-unknown-linux-gnuabi64 -C llvm-args=-mno-check-zero-division=0
//@[TRAP] needs-llvm-components: mips

#![crate_type = "lib"]
#![feature(no_core, intrinsics)]
#![no_core]

extern crate minicore;

#[rustc_intrinsic]
pub unsafe fn unchecked_div<T>(x: T, y: T) -> T;

#[rustc_intrinsic]
pub fn abort() -> !;

// NOTRAP-NOT: teq
// TRAP: teq
#[no_mangle]
pub fn div_i32(a: i32, b: i32) -> i32 {
    match a {
        0 => abort(),
        -1 => match b {
            -2147483648 => abort(),
            _ => unsafe { unchecked_div(a, b) },
        },
        _ => unsafe { unchecked_div(a, b) },
    }
}
