//@ only-x86_64
//@ assembly-output: emit-asm
//@ compile-flags: --crate-type=lib -Copt-level=3 -C target-cpu=x86-64-v4
//@ compile-flags: -C llvm-args=-x86-asm-syntax=intel

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
    // Even if we emit A+B, LLVM will sometimes reorder that to B+A, so this
    // test doesn't actually check which register is mov vs which is adc.

    // CHECK: mov [[TEMP1:.+]], qword ptr [{{rdx|rsi}} + 8*[[IND:.+]] + 8]
    // CHECK: adc [[TEMP1]], qword ptr [{{rdx|rsi}} + 8*[[IND]] + 8]
    // CHECK: mov qword ptr [rdi + 8*[[IND]] + 8], [[TEMP1]]
    // CHECK: mov [[TEMP2:.+]], qword ptr [{{rdx|rsi}} + 8*[[IND]] + 16]
    // CHECK: adc [[TEMP2]], qword ptr [{{rdx|rsi}} + 8*[[IND]] + 16]
    // CHECK: mov qword ptr [rdi + 8*[[IND]] + 16], [[TEMP2]]
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
