// Checks that the `carryless_mul` / `widening_carryless_mul` operations lower to
// the architectures' native carry-less (polynomial) multiply instructions.
//
// - Rust to LLVM IR: https://godbolt.org/z/sM914e4fo
// - LLVM IR to assembly: https://godbolt.org/z/5Y7naa4cY
//
//@revisions: x86_64 aarch64 riscv64
//@assembly-output: emit-asm
//@min-llvm-version: 23
//@compile-flags: -C opt-level=3
//
//@[x86_64] compile-flags: -C target-feature=+pclmulqdq,+avx2 -Cllvm-args=-x86-asm-syntax=intel
//@[x86_64] only-x86_64-unknown-linux-gnu
//
//@[aarch64] compile-flags: -C target-feature=+aes
//@[aarch64] only-aarch64-unknown-linux-gnu
//
//@[riscv64] compile-flags: -C target-feature=+zknd,+zbc
//@[riscv64] only-riscv64gc-unknown-linux-gnu

#![feature(uint_carryless_mul)]
#![crate_type = "lib"]

#[unsafe(no_mangle)]
fn carryless_mul_u8(a: u8, b: u8) -> u8 {
    // CHECK-LABEL: carryless_mul_u8:
    // x86_64: pclmulqdq
    // aarch64: pmul
    // riscv64: clmul
    a.carryless_mul(b)
}

#[unsafe(no_mangle)]
fn widening_carryless_mul_u8(a: u8, b: u8) -> u16 {
    // CHECK-LABEL: widening_carryless_mul_u8:
    // x86_64: pclmulqdq
    // aarch64: pmull
    // riscv64: clmul
    a.widening_carryless_mul(b)
}

#[unsafe(no_mangle)]
fn carryless_mul_u16(a: u16, b: u16) -> u16 {
    // CHECK-LABEL: carryless_mul_u16:
    // x86_64: pclmulqdq
    // aarch64: pmull
    // riscv64: clmul
    a.carryless_mul(b)
}

#[unsafe(no_mangle)]
fn widening_carryless_mul_u16(a: u16, b: u16) -> u32 {
    // CHECK-LABEL: widening_carryless_mul_u16:
    // x86_64: pclmulqdq
    // aarch64: pmull
    // riscv64: clmul
    a.widening_carryless_mul(b)
}

#[unsafe(no_mangle)]
fn carryless_mul_u32(a: u32, b: u32) -> u32 {
    // CHECK-LABEL: carryless_mul_u32:
    // x86_64: pclmulqdq
    // aarch64: pmull
    // riscv64: clmul
    a.carryless_mul(b)
}

#[unsafe(no_mangle)]
fn widening_carryless_mul_u32(a: u32, b: u32) -> u64 {
    // CHECK-LABEL: widening_carryless_mul_u32:
    // x86_64: pclmulqdq
    // aarch64: pmull
    // riscv64: slli
    // riscv64: slli
    // riscv64: clmulh
    a.widening_carryless_mul(b)
}

#[unsafe(no_mangle)]
fn carryless_mul_u64(a: u64, b: u64) -> u64 {
    // CHECK-LABEL: carryless_mul_u64:
    // x86_64: pclmulqdq
    // aarch64: pmull
    // riscv64: clmul
    a.carryless_mul(b)
}

#[unsafe(no_mangle)]
fn widening_carryless_mul_u64(a: u64, b: u64) -> u128 {
    // CHECK-LABEL: widening_carryless_mul_u64:
    //
    // x86_64: pclmulqdq
    // x86_64: vpextrq
    //
    // aarch64: rbit
    // aarch64: pmull
    // aarch64: pmull
    //
    // riscv64: clmul
    // riscv64: clmulh
    a.widening_carryless_mul(b)
}

#[unsafe(no_mangle)]
fn carryless_mul_u128(a: u128, b: u128) -> u128 {
    // CHECK-LABEL: carryless_mul_u128:
    //
    // x86_64: pclmulqdq
    // x86_64: pclmulqdq
    // x86_64: pclmulqdq
    // x86_64: xor
    //
    // aarch64: pmull
    // aarch64: pmull
    // aarch64: pmull
    // aarch64: eor
    //
    // riscv64: clmul
    // riscv64: clmul
    // riscv64: clmulh
    // riscv64: xor
    // riscv64: clmul
    // riscv64: xor
    a.carryless_mul(b)
}
