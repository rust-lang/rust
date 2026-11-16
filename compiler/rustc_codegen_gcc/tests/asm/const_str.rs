//@ assembly-output: emit-asm
//@ only-x86_64-unknown-linux-gnu
//@ compile-flags: -Copt-level=2

#![crate_type = "lib"]
#![allow(invalid_value)]

// The panic messages of the `assert_zero_valid` and `assert_inhabited` intrinsics are built with
// `const_str`, which passes the address of its global as the string data. The global must
// therefore hold the bytes themselves (not a pointer to a string literal), without a trailing
// NUL so that its size matches the length passed along, and it must be read-only.

// CHECK-LABEL: "zeroed_ref":
// CHECK: mov esi, 57
// CHECK: lea rdi, "[[REF_STR:str\.[0-9]+]]"[rip]
#[unsafe(no_mangle)]
pub fn zeroed_ref() -> &'static u8 {
    unsafe { std::mem::zeroed() }
}

// CHECK: .section .rodata.[[REF_STR]],"a"
// CHECK: .size "[[REF_STR]]", 57
// CHECK-NEXT: "[[REF_STR]]":
// CHECK-NEXT: .ascii "attempted to zero-initialize type `&u8`, which is invalid"
