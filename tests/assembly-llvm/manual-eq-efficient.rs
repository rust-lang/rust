// Regression test for #106269
//@ assembly-output: emit-asm
//@ compile-flags: --crate-type=lib -Copt-level=3 -C llvm-args=-x86-asm-syntax=intel
//@ only-x86_64
//@ ignore-sgx

pub struct S {
    a: u8,
    b: u8,
    c: u8,
    d: u8,
}

// CHECK-LABEL: manual_eq:
#[no_mangle]
pub fn manual_eq(s1: &S, s2: &S) -> bool {
    // CHECK: mov [[REG:[a-z0-9]+]], dword ptr [{{[a-z0-9]+}}]
    // CHECK-NEXT: cmp [[REG]], dword ptr [{{[a-z0-9]+}}]
    // CHECK-NEXT: sete al
    // CHECK: ret
    s1.a == s2.a && s1.b == s2.b && s1.c == s2.c && s1.d == s2.d
}
