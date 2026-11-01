//@ assembly-output: emit-asm
//@ compile-flags: -Copt-level=3
//@ only-x86_64

#![crate_type = "lib"]

type T = u8;
type T1 = (T, T, T, T, T, T, T, T);

// CHECK-LABEL: foo1a
// CHECK: cmpq
// CHECK-NEXT: sete
// CHECK-NEXT: {{retq|popq}}
#[no_mangle]
pub fn foo1a(a: T1, b: T1) -> bool {
    a == b
}

// CHECK-LABEL: foo1b
// CHECK: movq
// CHECK: cmpq
// CHECK-NEXT: sete
// CHECK-NEXT: {{retq|popq}}
#[no_mangle]
pub fn foo1b(a: &T1, b: &T1) -> bool {
    a == b
}
