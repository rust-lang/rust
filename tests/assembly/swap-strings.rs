// assembly-output: emit-asm
// compile-flags: --crate-type=lib -O -C llvm-args=-x86-asm-syntax=intel
// only-x86_64
// ignore-sgx
// ignore-debug

// Ensure that the swap uses SIMD registers and does not go to stack.

// CHECK-LABEL: swap_strings_xmm:
#[no_mangle]
pub fn swap_strings_xmm(a: &mut String, b: &mut String) {
    // CHECK-DAG: movups  [[A1:xmm.+]], xmmword ptr [[AX:.+]]
    // CHECK-DAG: mov     [[A2:r.+]], qword ptr [[AQ:.+]]
    // CHECK-DAG: movups  [[B1:xmm.+]], xmmword ptr [[BX:.+]]
    // CHECK-DAG: mov     [[B2:r.+]], qword ptr [[BQ:.+]]
    // CHECK-NOT: mov
    // CHECK-DAG: movups  xmmword ptr [[AX]], [[B1]]
    // CHECK-DAG: mov     qword ptr [[AQ]], [[B2]]
    // CHECK-DAG: movups  xmmword ptr [[BX]], [[A1]]
    // CHECK-DAG: mov     qword ptr [[BQ]], [[A2]]
    // CHECK: ret
    std::mem::swap(a, b);
}
