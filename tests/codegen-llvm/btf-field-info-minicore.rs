//@ add-minicore
//@ needs-llvm-components: bpf
//@ compile-flags: --target bpfel-unknown-none -Cdebuginfo=2

#![feature(allow_internal_unstable, btf_relocations, decl_macro, no_core)]
#![no_core]
#![no_std]
#![no_main]

extern crate minicore;
use minicore::*;

#[allow_internal_unstable(builtin_syntax)]
pub macro field_byte_offset($Container:ty, $($fields:expr)+ $(,)?) {{
    if builtin # btf_field_exists($Container, $($fields)+) {
        ::minicore::Option::Some(builtin # btf_field_byte_offset($Container, $($fields)+))
    } else {
        ::minicore::Option::None
    }
}}

#[allow_internal_unstable(builtin_syntax)]
pub macro field_byte_size($Container:ty, $($fields:expr)+ $(,)?) {{
    if builtin # btf_field_exists($Container, $($fields)+) {
        ::minicore::Option::Some(builtin # btf_field_byte_size($Container, $($fields)+))
    } else {
        ::minicore::Option::None
    }
}}

#[btf_relocatable]
#[repr(C)]
pub struct Inner {
    pub x: u32,
    pub y: u64,
}

#[btf_relocatable]
#[repr(C)]
pub union Payload {
    pub word: u64,
    pub half: u32,
}

#[btf_relocatable]
#[repr(C)]
pub struct Outer {
    pub pad: u32,
    pub inner: Inner,
    pub payload: Payload,
}

// Each `Option` query emits a `BPF_CORE_FIELD_EXISTS` relocation (kind 2), followed by either
// `BPF_CORE_FIELD_BYTE_OFFSET` (kind 0) or `BPF_CORE_FIELD_BYTE_SIZE` (kind 1). The value between
// the second and third colons is the compile-time fallback.
//
// CHECK-DAG: @"llvm.Outer:2:1$0:1" = external global i32, !llvm.preserve.access.index
// CHECK-DAG: @"llvm.Outer:0:8$0:1" = external global i32, !llvm.preserve.access.index
// CHECK-DAG: @"llvm.Outer:1:16$0:1" = external global i32, !llvm.preserve.access.index
//
// CHECK-DAG: @"llvm.Outer:2:1$0:1:1" = external global i32, !llvm.preserve.access.index
// CHECK-DAG: @"llvm.Outer:0:16$0:1:1" = external global i32, !llvm.preserve.access.index
// CHECK-DAG: @"llvm.Outer:1:8$0:1:1" = external global i32, !llvm.preserve.access.index
//
// CHECK-DAG: @"llvm.Outer:2:1$0:2:1" = external global i32, !llvm.preserve.access.index
// CHECK-DAG: @"llvm.Outer:0:24$0:2:1" = external global i32, !llvm.preserve.access.index
// CHECK-DAG: @"llvm.Outer:1:4$0:2:1" = external global i32, !llvm.preserve.access.index

// CHECK-LABEL: define{{.*}} @field_offset(
#[unsafe(no_mangle)]
pub fn field_offset() -> Option<usize> {
    field_byte_offset!(Outer, inner)
}

// CHECK-LABEL: define{{.*}} @field_size(
#[unsafe(no_mangle)]
pub fn field_size() -> Option<usize> {
    field_byte_size!(Outer, inner)
}

// CHECK-LABEL: define{{.*}} @nested_field_offset(
#[unsafe(no_mangle)]
pub fn nested_field_offset() -> Option<usize> {
    field_byte_offset!(Outer, inner.y)
}

// CHECK-LABEL: define{{.*}} @nested_field_size(
#[unsafe(no_mangle)]
pub fn nested_field_size() -> Option<usize> {
    field_byte_size!(Outer, inner.y)
}

// CHECK-LABEL: define{{.*}} @union_field_offset(
#[unsafe(no_mangle)]
pub fn union_field_offset() -> Option<usize> {
    field_byte_offset!(Outer, payload.half)
}

// CHECK-LABEL: define{{.*}} @union_field_size(
#[unsafe(no_mangle)]
pub fn union_field_size() -> Option<usize> {
    field_byte_size!(Outer, payload.half)
}
