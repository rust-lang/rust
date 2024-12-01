//@ only-x86_64
//@ assembly-output: emit-asm
//@ compile-flags: --crate-type=lib -O -C target-cpu=x86-64-v4
//@ compile-flags: -C llvm-args=-x86-asm-syntax=intel

#![no_std]
#![feature(bigint_helper_methods)]

// This checks that the `carrying_add` implementation successfully chains, to catch
// issues like <https://github.com/rust-lang/rust/issues/85532#issuecomment-2495119815>

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
    // CHECK: mov [[TEMP:r..]], qword ptr [rsi + 8*[[IND:r..]] + 8]
    // CHECK: adc [[TEMP]], qword ptr [rdx + 8*[[IND]] + 8]
    // CHECK: mov qword ptr [rdi + 8*[[IND]] + 8], [[TEMP]]
    // CHECK: mov [[TEMP]], qword ptr [rsi + 8*[[IND]] + 16]
    // CHECK: adc [[TEMP]], qword ptr [rdx + 8*[[IND]] + 16]
    // CHECK: mov qword ptr [rdi + 8*[[IND]] + 16], [[TEMP]]
    for i in 0..n {
        (*dest.add(i), carry) = u64::carrying_add(*src1.add(i), *src2.add(i), carry);
    }
    carry
}
