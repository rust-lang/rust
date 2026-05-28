//@ assembly-output: emit-asm
//@ compile-flags: --crate-type=lib -Copt-level=3 -C llvm-args=-x86-asm-syntax=intel
//@ only-x86_64
//@ ignore-sgx
//@ ignore-apple (manipulates rsp too)

// Depending on various codegen choices, this might end up copying
// a `<2 x i8>`, an `i16`, or two `i8`s.
// Regardless of those choices, make sure the instructions use (2-byte) words.

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
