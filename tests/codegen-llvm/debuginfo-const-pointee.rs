// Verify that the pointee of `&T` and `*const T` is `const`-qualified in debuginfo, also for the
// data pointer of wide pointers, while `&mut T`, `*mut T` and `Box<T>` are not. Interior mutability
// is ignored. All of them remain `DW_TAG_pointer_type`s.

//@ compile-flags: -C debuginfo=2 -Copt-level=0
//@ ignore-msvc the types are named differently

#![crate_type = "lib"]

use std::cell::Cell;

pub trait Trait {}

// CHECK-DAG: ![[U32:[0-9]+]] = !DIBasicType(name: "u32"
// CHECK-DAG: ![[CONST_U32:[0-9]+]] = !DIDerivedType(tag: DW_TAG_const_type, baseType: ![[U32]])
// CHECK-DAG: !DIDerivedType(tag: DW_TAG_pointer_type, name: "&u32", baseType: ![[CONST_U32]]
// CHECK-DAG: !DIDerivedType(tag: DW_TAG_pointer_type, name: "&mut u32", baseType: ![[U32]]
// CHECK-DAG: !DIDerivedType(tag: DW_TAG_pointer_type, name: "*const u32", baseType: ![[CONST_U32]]
// CHECK-DAG: !DIDerivedType(tag: DW_TAG_pointer_type, name: "*mut u32", baseType: ![[U32]]
// CHECK-DAG: !DIDerivedType(tag: DW_TAG_pointer_type, name: "alloc::boxed::Box<u32, alloc::alloc::Global>", baseType: ![[U32]]
// CHECK-DAG: ![[CELL:[0-9]+]] = !DICompositeType(tag: DW_TAG_structure_type, name: "Cell<u32>"
// CHECK-DAG: ![[CONST_CELL:[0-9]+]] = !DIDerivedType(tag: DW_TAG_const_type, baseType: ![[CELL]])
// CHECK-DAG: !DIDerivedType(tag: DW_TAG_pointer_type, name: "&core::cell::Cell<u32>", baseType: ![[CONST_CELL]]
#[no_mangle]
pub fn thin(_a: &u32, _b: &mut u32, _c: *const u32, _d: *mut u32, _e: Box<u32>, _f: &Cell<u32>) {}

// CHECK-DAG: ![[U8:[0-9]+]] = !DIBasicType(name: "u8"
// CHECK-DAG: ![[CONST_U8:[0-9]+]] = !DIDerivedType(tag: DW_TAG_const_type, baseType: ![[U8]])

// CHECK-DAG: ![[SLICE:[0-9]+]] = !DICompositeType(tag: DW_TAG_structure_type, name: "&[u8]", {{.*}}elements: ![[SLICE_ELEMS:[0-9]+]]
// CHECK-DAG: ![[SLICE_ELEMS]] = !{![[SLICE_DATA:[0-9]+]], ![[SLICE_LEN:[0-9]+]]}
// CHECK-DAG: ![[SLICE_DATA]] = !DIDerivedType(tag: DW_TAG_member, name: "data_ptr", scope: ![[SLICE]], {{.*}}baseType: ![[SLICE_PTR:[0-9]+]]
// CHECK-DAG: ![[SLICE_PTR]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[CONST_U8]]

// CHECK-DAG: ![[MUT_SLICE:[0-9]+]] = !DICompositeType(tag: DW_TAG_structure_type, name: "&mut [u8]", {{.*}}elements: ![[MUT_SLICE_ELEMS:[0-9]+]]
// CHECK-DAG: ![[MUT_SLICE_ELEMS]] = !{![[MUT_SLICE_DATA:[0-9]+]], ![[MUT_SLICE_LEN:[0-9]+]]}
// CHECK-DAG: ![[MUT_SLICE_DATA]] = !DIDerivedType(tag: DW_TAG_member, name: "data_ptr", scope: ![[MUT_SLICE]], {{.*}}baseType: ![[MUT_SLICE_PTR:[0-9]+]]
// CHECK-DAG: ![[MUT_SLICE_PTR]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[U8]]

// CHECK-DAG: ![[RAW_SLICE:[0-9]+]] = !DICompositeType(tag: DW_TAG_structure_type, name: "*const [u8]", {{.*}}elements: ![[RAW_SLICE_ELEMS:[0-9]+]]
// CHECK-DAG: ![[RAW_SLICE_ELEMS]] = !{![[RAW_SLICE_DATA:[0-9]+]], ![[RAW_SLICE_LEN:[0-9]+]]}
// The data pointer is the same (uniqued) node as for `&[u8]`.
// CHECK-DAG: ![[RAW_SLICE_DATA]] = !DIDerivedType(tag: DW_TAG_member, name: "data_ptr", scope: ![[RAW_SLICE]], {{.*}}baseType: ![[SLICE_PTR]]

// CHECK-DAG: ![[DYN:[0-9]+]] = !DICompositeType(tag: DW_TAG_structure_type, name: "&dyn {{.*}}Trait", {{.*}}elements: ![[DYN_ELEMS:[0-9]+]]
// CHECK-DAG: ![[DYN_ELEMS]] = !{![[DYN_DATA:[0-9]+]], ![[DYN_VTABLE:[0-9]+]]}
// CHECK-DAG: ![[DYN_DATA]] = !DIDerivedType(tag: DW_TAG_member, name: "pointer", scope: ![[DYN]], {{.*}}baseType: ![[DYN_PTR:[0-9]+]]
// CHECK-DAG: ![[DYN_PTR]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[CONST_DYN:[0-9]+]]
// CHECK-DAG: ![[CONST_DYN]] = !DIDerivedType(tag: DW_TAG_const_type, baseType:
#[no_mangle]
pub fn wide(_a: &[u8], _b: &mut [u8], _c: *const [u8], _d: &dyn Trait) {}
