// MIPS assembler uses the label prefix `$anon.` for local anonymous variables
// other architectures (including ARM and x86-64) use the prefix `.Lanon.`
//@ only-linux
//@ assembly-output: emit-asm
//@ compile-flags: --crate-type=lib -Copt-level=3 -Cllvm-args=-enable-global-merge=0
//@ edition: 2024

use std::ffi::CStr;

// CHECK: .section .rodata.str1.{{[12]}},"aMS"
// CHECK: {{(\.L|\$)}}anon.{{.+}}:
// CHECK-NEXT: .asciz "foo"
#[unsafe(no_mangle)]
static CSTR: &[u8; 4] = b"foo\0";

// CHECK-NOT: .section
// CHECK: {{(\.L|\$)}}anon.{{.+}}:
// CHECK-NEXT: .asciz "bar"
#[unsafe(no_mangle)]
pub fn cstr() -> &'static CStr {
    c"bar"
}

// CHECK-NOT: .section
// CHECK: {{(\.L|\$)}}anon.{{.+}}:
// CHECK-NEXT: .asciz "baz"
#[unsafe(no_mangle)]
pub fn manual_cstr() -> &'static str {
    "baz\0"
}
