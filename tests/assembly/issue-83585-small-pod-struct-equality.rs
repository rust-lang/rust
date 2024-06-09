//@ assembly-output: emit-asm
//@ compile-flags:-Copt-level=3
//@ only-x86_64

#![crate_type = "lib"]

type T = u8;
type T1 = (T, T, T, T, T, T, T, T);
type T2 = [T; 8];

// CHECK-LABEL: foo1a
// CHECK: cmp
// CHECK-NEXT: sete
// CHECK-NEXT: ret
#[no_mangle]
pub fn foo1a(a: T1, b: T1) -> bool {
    a == b
}

// CHECK-LABEL: foo1b
// CHECK: mov
// CHECK-NEXT: cmp
// CHECK-NEXT: sete
// CHECK-NEXT: ret
#[no_mangle]
pub fn foo1b(a: &T1, b: &T1) -> bool {
    a == b
}
