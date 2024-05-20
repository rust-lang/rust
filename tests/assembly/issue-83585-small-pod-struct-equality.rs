//@ assembly-output: emit-asm
//@ compile-flags:-Copt-level=3
//@ only-x86_64

#![crate_type = "lib"]

#[no_mangle]
type T = u8;
type T1 = (T, T, T, T, T, T, T, T);
type T2 = [T; 8];

#[no_mangle]
// CHECK-LABEL: foo1a
// CHECK: cmp
// CHECK-NEXT: set
// CHECK-NEXT: ret
pub fn foo1a(a: T1, b: T1) -> bool {
    a == b
}

#[no_mangle]
// CHECK-LABEL: foo1b
// CHECK: mov
// CHECK-NEXT: cmp
// CHECK-NEXT: set
// CHECK-NEXT: ret
pub fn foo1b(a: &T1, b: &T1) -> bool {
    a == b
}

