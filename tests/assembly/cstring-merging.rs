//@ only-linux
//@ assembly-output: emit-asm
//@ compile-flags: --crate-type=lib -Copt-level=3
//@ edition: 2024

use std::ffi::CStr;

// CHECK: .section .rodata.str1.{{[12]}},"aMS"
// CHECK: .Lanon.{{.+}}:
// CHECK-NEXT: .asciz "foo"
#[unsafe(no_mangle)]
static CSTR: &[u8; 4] = b"foo\0";

// CHECK-NOT: .section
// CHECK: .Lanon.{{.+}}:
// CHECK-NEXT: .asciz "bar"
#[unsafe(no_mangle)]
pub fn cstr() -> &'static CStr {
    c"bar"
}

// CHECK-NOT: .section
// CHECK: .Lanon.{{.+}}:
// CHECK-NEXT: .asciz "baz"
#[unsafe(no_mangle)]
pub fn manual_cstr() -> &'static str {
    "baz\0"
}
