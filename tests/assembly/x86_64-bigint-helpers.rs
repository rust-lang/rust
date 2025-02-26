//@ only-x86_64
//@ assembly-output: emit-asm
//@ compile-flags: --crate-type=lib -Copt-level=3 -C target-cpu=x86-64-v4
//@ compile-flags: -C llvm-args=-x86-asm-syntax=intel
//@ revisions: llvm-pre-20 llvm-20
//@ [llvm-20] min-llvm-version: 20
//@ [llvm-pre-20] max-llvm-major-version: 19

#![no_std]
#![feature(bigint_helper_methods)]

// This checks that the `carrying_add` and `borrowing_sub` implementation successfully chain,
// to catch issues like <https://github.com/rust-lang/rust/issues/85532#issuecomment-2495119815>

// This forces the ABI to avoid the windows-vs-linux ABI differences.

// CHECK-LABEL: bigint_chain_carrying_add:
#[no_mangle]
pub unsafe extern "sysv64" fn bigint_chain_carrying_add(
    dest: *mut u64,
    src1: *const u64,
    src2: *const u64,
    n: usize,
    mut carry: bool,
) -> bool {
    // llvm-pre-20: mov [[TEMP:r..]], qword ptr [rsi + 8*[[IND:r..]] + 8]
    // llvm-pre-20: adc [[TEMP]], qword ptr [rdx + 8*[[IND]] + 8]
    // llvm-pre-20: mov qword ptr [rdi + 8*[[IND]] + 8], [[TEMP]]
    // llvm-pre-20: mov [[TEMP]], qword ptr [rsi + 8*[[IND]] + 16]
    // llvm-pre-20: adc [[TEMP]], qword ptr [rdx + 8*[[IND]] + 16]
    // llvm-pre-20: mov qword ptr [rdi + 8*[[IND]] + 16], [[TEMP]]
    // llvm-20: adc [[TEMP:r..]], qword ptr [rdx + 8*[[IND:r..]]]
    // llvm-20: mov qword ptr [rdi + 8*[[IND]]], [[TEMP]]
    // llvm-20: mov [[TEMP]], qword ptr [rsi + 8*[[IND]] + 8]
    // llvm-20: adc [[TEMP]], qword ptr [rdx + 8*[[IND]] + 8]
    for i in 0..n {
        (*dest.add(i), carry) = u64::carrying_add(*src1.add(i), *src2.add(i), carry);
    }
    carry
}

// CHECK-LABEL: bigint_chain_borrowing_sub:
#[no_mangle]
pub unsafe extern "sysv64" fn bigint_chain_borrowing_sub(
    dest: *mut u64,
    src1: *const u64,
    src2: *const u64,
    n: usize,
    mut carry: bool,
) -> bool {
    // CHECK: mov [[TEMP:r..]], qword ptr [rsi + 8*[[IND:r..]] + 8]
    // CHECK: sbb [[TEMP]], qword ptr [rdx + 8*[[IND]] + 8]
    // CHECK: mov qword ptr [rdi + 8*[[IND]] + 8], [[TEMP]]
    // CHECK: mov [[TEMP]], qword ptr [rsi + 8*[[IND]] + 16]
    // CHECK: sbb [[TEMP]], qword ptr [rdx + 8*[[IND]] + 16]
    // CHECK: mov qword ptr [rdi + 8*[[IND]] + 16], [[TEMP]]
    for i in 0..n {
        (*dest.add(i), carry) = u64::borrowing_sub(*src1.add(i), *src2.add(i), carry);
    }
    carry
}
