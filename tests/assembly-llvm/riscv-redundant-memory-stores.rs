//! Regression test for <https://github.com/rust-lang/rust/issues/149762>:

//@ assembly-output: emit-asm
//@ compile-flags: -Copt-level=3 --target riscv64gc-unknown-linux-gnu
//@ needs-llvm-components: riscv
//@ only-riscv64

pub struct SomeComplexType {
    a: u64,
    b: u64,
    c: u64,
}

// CHECK-LABEL: with_mut_param
#[no_mangle]
pub fn with_mut_param(mut a: SomeComplexType) -> SomeComplexType {
    // CHECK: ld a2, 0(a1)
    // CHECK-NEXT: ld a3, 8(a1)
    // CHECK-NEXT: ld a4, 16(a1)
    // CHECK-NEXT: addi a2, a2, 10
    // CHECK-NEXT: addi a3, a3, 2
    // CHECK-NEXT: sd a2, 0(a1)
    // CHECK-NEXT: sd a3, 8(a1)
    // CHECK-NEXT: sd a2, 0(a0)
    // CHECK-NEXT: sd a3, 8(a0)
    // CHECK-NEXT: sd a4, 16(a0)
    // CHECK-NEXT: ret
    a.a += 10;
    a.b += 2;
    a
}

fn main() {}
