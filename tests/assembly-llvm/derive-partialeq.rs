//@ compile-flags: -Copt-level=3
//@ assembly-output: emit-asm
//@ only-x86_64

#![crate_type = "lib"]

#[derive(PartialEq)]
struct Foo {
    a: u16,
    b: u16,
}

// CHECK-LABEL: two_u16:
#[no_mangle]
pub fn two_u16(a: &Foo, b: &Foo) -> bool {
    // CHECK-NEXT: .cfi_startproc
    // CHECK-NEXT: movl
    // CHECK-NEXT: cmpl
    // CHECK-NEXT: sete
    // CHECK-NEXT: retq
    a == b
}
