//! This file tests that we correctly generate GEP instructions for DST
//! field offsets.
//@ compile-flags: -C no-prepopulate-passes -Copt-level=0

#![crate_type = "lib"]
#![feature(extern_types, sized_hierarchy)]

use std::marker::PointeeSized;
use std::ptr::addr_of;

// Hack to get the correct type for usize
// CHECK: @helper([[USIZE:i[0-9]+]] %_1)
#[no_mangle]
pub fn helper(_: usize) {}

struct Dst<T: PointeeSized> {
    x: u32,
    y: u8,
    z: T,
}

// CHECK: @dst_dyn_trait_offset(ptr align {{[0-9]+}} [[DATA_PTR:%.+]], ptr align {{[0-9]+}} [[VTABLE_PTR:%.+]])
#[no_mangle]
pub fn dst_dyn_trait_offset(s: &Dst<dyn Drop>) -> &dyn Drop {
    // The alignment of dyn trait is unknown, so we compute the offset based on align from the
    // vtable.

    // CHECK: [[SIZE_PTR:%[0-9]+]] = getelementptr inbounds i8, ptr [[VTABLE_PTR]]
    // CHECK: load [[USIZE]], ptr [[SIZE_PTR]]
    // CHECK: [[ALIGN_PTR:%[0-9]+]] = getelementptr inbounds i8, ptr [[VTABLE_PTR]]
    // CHECK: load [[USIZE]], ptr [[ALIGN_PTR]]

    // CHECK: getelementptr inbounds i8, ptr [[DATA_PTR]]
    // CHECK-NEXT: insertvalue
    // CHECK-NEXT: insertvalue
    // CHECK-NEXT: ret
    &s.z
}

// CHECK-LABEL: @dst_slice_offset
#[no_mangle]
pub fn dst_slice_offset(s: &Dst<[u16]>) -> &[u16] {
    // The alignment of [u16] is known, so we generate a GEP directly.

    // CHECK: start:
    // CHECK-NEXT: getelementptr inbounds i8, {{.+}}, [[USIZE]] 6
    // CHECK-NEXT: insertvalue
    // CHECK-NEXT: insertvalue
    // CHECK-NEXT: ret
    &s.z
}

#[repr(packed)]
struct PackedDstSlice {
    x: u32,
    y: u8,
    z: [u16],
}

// CHECK-LABEL: @packed_dst_slice_offset
#[no_mangle]
pub fn packed_dst_slice_offset(s: &PackedDstSlice) -> *const [u16] {
    // The alignment of [u16] is known, so we generate a GEP directly.

    // CHECK: start:
    // CHECK-NEXT: getelementptr inbounds i8, {{.+}}, [[USIZE]] 5
    // CHECK-NEXT: insertvalue
    // CHECK-NEXT: insertvalue
    // CHECK-NEXT: ret
    addr_of!(s.z)
}

extern "C" {
    pub type Extern;
}

// CHECK-LABEL: @dst_extern
#[no_mangle]
pub fn dst_extern(s: &Dst<Extern>) -> &Extern {
    // Computing the alignment of an extern type is currently unsupported and just panics.

    // CHECK: call void @{{.+}}panic
    &s.z
}
