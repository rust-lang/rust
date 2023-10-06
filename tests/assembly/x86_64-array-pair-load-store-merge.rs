// assembly-output: emit-asm
// compile-flags: --crate-type=lib -O -C llvm-args=-x86-asm-syntax=intel
// only-x86_64
// ignore-sgx

// This is emitted in a way that results in two loads and two stores in LLVM-IR.
// Confirm that that doesn't mean 4 instructions in assembly.

// CHECK-LABEL: array_copy_2_elements:
#[no_mangle]
pub fn array_copy_2_elements(a: &[u8; 2], p: &mut [u8; 2]) {
    // CHECK-NOT: byte
    // CHECK-NOT: mov
    // CHECK: mov{{.+}}, word ptr
    // CHECK-NEXT: mov word ptr
    // CHECK-NEXT: ret
    *p = *a;
}
