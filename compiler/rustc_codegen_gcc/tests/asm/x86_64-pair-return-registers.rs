//@ assembly-output: emit-asm
//@ only-x86_64-unknown-linux-gnu
//@ compile-flags: -Copt-level=2

// cg_llvm returns each scalar of a pair in its own register, even when the pair fits in one. A
// backend that packs both scalars into one register, like GCC does for a struct that fits in an
// eightbyte, exchanges garbage with code built by cg_llvm.

#![feature(f16)]
#![crate_type = "lib"]

// CHECK-LABEL: {{"?}}pair_u32{{"?}}:
// CHECK-NOT: sal
// CHECK: rdx
// CHECK: ret
#[unsafe(no_mangle)]
pub fn pair_u32(first: u32, second: u32) -> (u32, u32) {
    (first, second)
}

unsafe extern "Rust" {
    safe fn external_pair_u32() -> (u32, u32);
}

// CHECK-LABEL: {{"?}}second_of_pair{{"?}}:
// CHECK: call
// CHECK-NOT: shr
// CHECK: eax, edx
// CHECK: ret
#[unsafe(no_mangle)]
pub fn second_of_pair() -> u32 {
    external_pair_u32().1
}

// CHECK-LABEL: {{"?}}swap_f16{{"?}}:
// CHECK: xmm1,
// CHECK: ret
#[unsafe(no_mangle)]
pub fn swap_f16(first: f16, second: f16) -> (f16, f16) {
    (second, first)
}
