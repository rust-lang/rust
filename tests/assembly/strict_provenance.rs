// assembly-output: emit-asm
// compile-flags: -Copt-level=1
// only-x86_64
// ignore-sgx
// min-llvm-version: 15.0
#![crate_type = "rlib"]

// CHECK-LABEL: old_style
// CHECK: movq %{{.*}}, %rax
// CHECK: orq $1, %rax
// CHECK: retq
#[no_mangle]
pub fn old_style(a: *mut u8) -> *mut u8 {
    (a as usize | 1) as *mut u8
}

// CHECK-LABEL: cheri_compat
// CHECK: movq %{{.*}}, %rax
// CHECK: orq $1, %rax
// CHECK: retq
#[no_mangle]
pub fn cheri_compat(a: *mut u8) -> *mut u8 {
    let old = a as usize;
    let new = old | 1;
    let diff = new.wrapping_sub(old);
    a.wrapping_add(diff)
}

// CHECK-LABEL: definitely_not_a_null_pointer
// CHECK: movq %{{.*}}, %rax
// CHECK: orq $1, %rax
// CHECK: retq
#[no_mangle]
pub fn definitely_not_a_null_pointer(a: *mut u8) -> *mut u8 {
    let old = a as usize;
    let new = old | 1;
    a.wrapping_sub(old).wrapping_add(new)
}
