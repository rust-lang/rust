// MIPS assembler uses the label prefix `$`.
// Other architectures (including ARM and x86-64) use the prefix `.L`.
// Hexagon uses `.string` instead of `.asciz` for null-terminated strings
//@ only-linux
//@ assembly-output: emit-asm
//@ compile-flags: --crate-type=lib -Copt-level=3 -Cllvm-args=-enable-global-merge=0
//@ edition: 2024

use std::ffi::CStr;

// CHECK: .section .rodata.str1.{{[12]}},"aMS"
// CHECK: {{(\.L|\$)}}__unnamed_{{[0-9]+}}:
// CHECK-NEXT: .{{asciz|string}} "foo"
#[unsafe(no_mangle)]
static CSTR: &[u8; 4] = b"foo\0";

// CHECK: .section .rodata.str1.{{[12]}},"aMS"
// CHECK: {{(\.L|\$)}}__unnamed_{{[0-9]+}}:
// CHECK-NEXT: .{{asciz|string}} "bar"
#[unsafe(no_mangle)]
pub fn cstr() -> &'static CStr {
    c"bar"
}

// CHECK-NOT: .section
// CHECK: {{(\.L|\$)}}__unnamed_{{[0-9]+}}:
// CHECK-NEXT: .{{asciz|string}} "baz"
#[unsafe(no_mangle)]
pub fn manual_cstr() -> &'static str {
    "baz\0"
}
