// Regression test for #56057
//@ assembly-output: emit-asm
//@ compile-flags: --crate-type=lib -Copt-level=3 -C llvm-args=-x86-asm-syntax=intel
//@ only-x86_64
//@ ignore-sgx

// The masked bits are already the byte offset of the element, so the index
// should be used directly without shifting it down and up again.

// CHECK-LABEL: masked_index:
#[no_mangle]
pub fn masked_index(table: &[u128; 4], idx: i32) -> u128 {
    // CHECK-NOT: shr
    // CHECK-NOT: shl
    // CHECK: and {{[a-z0-9]+}}, 48
    // CHECK-NOT: shr
    // CHECK-NOT: shl
    // CHECK: ret
    table[(idx as usize & 0b11_0000) >> 4]
}
