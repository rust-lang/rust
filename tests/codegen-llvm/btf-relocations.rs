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
    builtin # btf_field_byte_offset($Container, $($fields)+)
}}

#[allow_internal_unstable(builtin_syntax)]
pub macro field_byte_size($Container:ty, $($fields:expr)+ $(,)?) {{
    builtin # btf_field_byte_size($Container, $($fields)+)
}}

#[allow_internal_unstable(builtin_syntax)]
pub macro field_exists($Container:ty, $($fields:expr)+ $(,)?) {{
    builtin # btf_field_exists($Container, $($fields)+)
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

// BTF CO-RE relocations are represented in the following format in LLVM IR:
//
// llvm.<root type>:<relocation kind>:<compile-time value>$<access path>
//
// For example:
//
// llvm.Outer:0:8$0:1
//      │     │ │  └─ access path: 0:1 (base 0, field 1 `inner`), Outer.inner
//      │     │ └──── compile-time value: 8
//      │     └────── relocation kind: 0 (FIELD_BYTE_OFFSET)
//      └──────────── root type: Outer
//
// llvm.Outer:1:8$0:1:1
//      │     │ │  └─ access path: 0:1:1 (base 0, field 1 `inner`, field 1 `y`), Outer.inner.y
//      │     │ └──── compile-time value: 8
//      │     └────── relocation kind: 1(FIELD_BYTE_SIZE)
//      └──────────── root type: Outer
//
// The relocation types are standardized across Linux kernel[0] and LLVM[1].
// Rust currently supports:
//
// * FIELD_BYTE_OFFSET: 0
// * FIELD_BYTE_SIZE: 1
// * FIELD_EXISTS: 2
//
// [0] https://elixir.bootlin.com/linux/v7.2.5/source/include/uapi/linux/bpf.h#L7616
// [1] https://github.com/llvm/llvm-project/blob/llvmorg-23.1.1/llvm/include/llvm/DebugInfo/BTF/BTF.h#L281
//
// CHECK-DAG: @"llvm.Outer:[[BTF_FIELD_BYTE_OFFSET:0]]:[[INNER_OFFSET:8]]$[[INNER_PATH:0:1]]" = external global i32, !llvm.preserve.access.index !0 #0
// CHECK-DAG: @"llvm.Outer:[[BTF_FIELD_BYTE_SIZE:1]]:[[INNER_SIZE:16]]$[[INNER_PATH]]" = external global i32, !llvm.preserve.access.index !0 #0
// CHECK-DAG: @"llvm.Outer:[[BTF_FIELD_EXISTS:2]]:[[INNER_EXISTS:1]]$[[INNER_PATH]]" = external global i32, !llvm.preserve.access.index !0 #0
//
// CHECK-DAG: @"llvm.Outer:[[BTF_FIELD_BYTE_OFFSET]]:[[INNER_Y_OFFSET:16]]$[[INNER_Y_PATH:0:1:1]]" = external global i32, !llvm.preserve.access.index !0 #0
// CHECK-DAG: @"llvm.Outer:[[BTF_FIELD_BYTE_SIZE]]:[[INNER_Y_SIZE:8]]$[[INNER_Y_PATH]]" = external global i32, !llvm.preserve.access.index !0 #0
// CHECK-DAG: @"llvm.Outer:[[BTF_FIELD_EXISTS]]:[[INNER_Y_EXISTS:1]]$[[INNER_Y_PATH]]" = external global i32, !llvm.preserve.access.index !0 #0
//
// CHECK-DAG: @"llvm.Outer:[[BTF_FIELD_BYTE_OFFSET]]:[[PAYLOAD_HALF_OFFSET:24]]$[[PAYLOAD_HALF_PATH:0:2:1]]" = external global i32, !llvm.preserve.access.index !0 #0
// CHECK-DAG: @"llvm.Outer:[[BTF_FIELD_BYTE_SIZE]]:[[PAYLOAD_HALF_SIZE:4]]$[[PAYLOAD_HALF_PATH]]" = external global i32, !llvm.preserve.access.index !0 #0
// CHECK-DAG: @"llvm.Outer:[[BTF_FIELD_EXISTS]]:[[PAYLOAD_HALF_EXISTS:1]]$[[PAYLOAD_HALF_PATH]]" = external global i32, !llvm.preserve.access.index !0 #0

// CHECK-LABEL: define{{.*}} @field_offset(
#[unsafe(no_mangle)]
pub fn field_offset() -> usize {
    // CHECK: [[A:%.*]] = load i32, ptr @"llvm.Outer:[[BTF_FIELD_BYTE_OFFSET]]:[[INNER_OFFSET]]$[[INNER_PATH]]", align 4
    // CHECK-NEXT: [[B:%.*]] = tail call i32 @llvm.bpf.passthrough.i32.i32(i32 {{[0-9]+}}, i32 [[A]])
    // CHECK-NEXT: [[C:%.*]] = zext i32 [[B]] to i64
    // CHECK-NEXT: ret i64 [[C]]
    field_byte_offset!(Outer, inner)
}

// CHECK-LABEL: define{{.*}} @field_size(
#[unsafe(no_mangle)]
pub fn field_size() -> usize {
    // CHECK: [[A:%.*]] = load i32, ptr @"llvm.Outer:[[BTF_FIELD_BYTE_SIZE]]:[[INNER_SIZE]]$[[INNER_PATH]]", align 4
    // CHECK-NEXT: [[B:%.*]] = tail call i32 @llvm.bpf.passthrough.i32.i32(i32 {{[0-9]+}}, i32 [[A]])
    // CHECK-NEXT: [[C:%.*]] = zext i32 [[B]] to i64
    // CHECK-NEXT: ret i64 [[C]]
    field_byte_size!(Outer, inner)
}

// CHECK-LABEL: define{{.*}} @field_exists(
#[unsafe(no_mangle)]
pub fn field_exists() -> bool {
    // CHECK: [[A:%.*]] = load i32, ptr @"llvm.Outer:[[BTF_FIELD_EXISTS]]:[[INNER_EXISTS]]$[[INNER_PATH]]", align 4
    // CHECK-NEXT: [[B:%.*]] = tail call i32 @llvm.bpf.passthrough.i32.i32(i32 {{[0-9]+}}, i32 [[A]])
    // CHECK-NEXT: [[C:%.*]] = icmp ne i32 [[B]], 0
    // CHECK-NEXT: ret i1 [[C]]
    field_exists!(Outer, inner)
}

// CHECK-LABEL: define{{.*}} @nested_field_offset(
#[unsafe(no_mangle)]
pub fn nested_field_offset() -> usize {
    // CHECK: [[A:%.*]] = load i32, ptr @"llvm.Outer:[[BTF_FIELD_BYTE_OFFSET]]:[[INNER_Y_OFFSET]]$[[INNER_Y_PATH]]", align 4
    // CHECK-NEXT: [[B:%.*]] = tail call i32 @llvm.bpf.passthrough.i32.i32(i32 {{[0-9]+}}, i32 [[A]])
    // CHECK-NEXT: [[C:%.*]] = zext i32 %1 to i64
    // CHECK-NEXT: ret i64 [[C]]
    field_byte_offset!(Outer, inner.y)
}

// CHECK-LABEL: define{{.*}} @nested_field_size(
#[unsafe(no_mangle)]
pub fn nested_field_size() -> usize {
    // CHECK: [[A:%.*]] = load i32, ptr @"llvm.Outer:[[BTF_FIELD_BYTE_SIZE]]:[[INNER_Y_SIZE]]$[[INNER_Y_PATH]]", align 4
    // CHECK-NEXT: [[B:%.*]] = tail call i32 @llvm.bpf.passthrough.i32.i32(i32 {{[0-9]+}}, i32 [[A]])
    // CHECK-NEXT: [[C:%.*]] = zext i32 [[B]] to i64
    // CHECK-NEXT: ret i64 [[C]]
    field_byte_size!(Outer, inner.y)
}

// CHECK-LABEL: define{{.*}} @nested_field_exists(
#[unsafe(no_mangle)]
pub fn nested_field_exists() -> bool {
    // CHECK: [[A:%.*]] = load i32, ptr @"llvm.Outer:[[BTF_FIELD_EXISTS]]:[[INNER_Y_EXISTS]]$[[INNER_Y_PATH]]", align 4
    // CHECK-NEXT: [[B:%.*]] = tail call i32 @llvm.bpf.passthrough.i32.i32(i32 {{[0-9]+}}, i32 [[A]])
    // CHECK-NEXT: [[C:%.*]] = icmp ne i32 [[B]], 0
    // CHECK-NEXT: ret i1 [[C]]
    field_exists!(Outer, inner.y)
}

// CHECK-LABEL: define{{.*}} @union_field_offset(
#[unsafe(no_mangle)]
pub fn union_field_offset() -> usize {
    // CHECK: [[A:%.*]] = load i32, ptr @"llvm.Outer:[[BTF_FIELD_BYTE_OFFSET]]:[[PAYLOAD_HALF_OFFSET]]$[[PAYLOAD_HALF_PATH]]", align 4
    // CHECK-NEXT: [[B:%.*]] = tail call i32 @llvm.bpf.passthrough.i32.i32(i32 {{[0-9]+}}, i32 [[A]])
    // CHECK-NEXT: [[C:%.*]] = zext i32 [[B]] to i64
    // CHECK-NEXT: ret i64 [[C]]
    field_byte_offset!(Outer, payload.half)
}

// CHECK-LABEL: define{{.*}} @union_field_size(
#[unsafe(no_mangle)]
pub fn union_field_size() -> usize {
    // CHECK: [[A:%.*]] = load i32, ptr @"llvm.Outer:[[BTF_FIELD_BYTE_SIZE]]:[[PAYLOAD_HALF_SIZE]]$[[PAYLOAD_HALF_PATH]]", align 4
    // CHECK-NEXT: [[B:%.*]] = tail call i32 @llvm.bpf.passthrough.i32.i32(i32 {{[0-9]+}}, i32 [[A]])
    // CHECK-NEXT: [[C:%.*]] = zext i32 [[B]] to i64
    // CHECK-NEXT: ret i64 [[C]]
    field_byte_size!(Outer, payload.half)
}

// CHECK-LABEL: define{{.*}} @union_field_exists(
#[unsafe(no_mangle)]
pub fn union_field_exists() -> bool {
    // CHECK: [[A:%.*]] = load i32, ptr @"llvm.Outer:[[BTF_FIELD_EXISTS]]:[[PAYLOAD_HALF_EXISTS]]$[[PAYLOAD_HALF_PATH]]", align 4
    // CHECK-NEXT: [[B:%.*]] = tail call i32 @llvm.bpf.passthrough.i32.i32(i32 {{[0-9]+}}, i32 [[A]])
    // CHECK-NEXT: [[C:%.*]] = icmp ne i32 [[B]], 0
    // CHECK-NEXT: ret i1 [[C]]
    field_exists!(Outer, payload.half)
}
