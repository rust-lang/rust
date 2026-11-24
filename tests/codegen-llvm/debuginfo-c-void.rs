//@ compile-flags:-g -C opt-level=0 -C panic=abort

// Check that the base types of the pointer types of `core::ffi::c_void`
// (`*const c_void`, `*mut c_void`) are NULL, as expected by LLVM:
// https://llvm.org/docs/LangRef.html#diderivedtype

#![crate_type = "lib"]

// CHECK-NOT: !DICompositeType(tag: DW_TAG_enumeration_type, name: "c_void"
// CHECK-NOT: !DIEnumerator(name: "__variant1", value: 0, isUnsigned: true)
// CHECK-NOT: !DIEnumerator(name: "__variant2", value: 1, isUnsigned: true)
use std::ffi::c_void;

// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "Foo", {{.*}}elements: ![[ELEMENTS:[0-9]+]]
// CHECK: ![[ELEMENTS]] = !{![[A:[0-9]+]], ![[B:[0-9]+]]}
pub struct Foo {
    // CHECK: ![[A]] = !DIDerivedType(tag: DW_TAG_member, name: "a", {{.*}}baseType: ![[A_BASE_TYPE:[0-9]+]]
    // CHECK: ![[A_BASE_TYPE]] = !DIDerivedType(tag: DW_TAG_pointer_type, name: "{{.*}}core::ffi::c_void{{.*}}", {{.*}}baseType: null
    a: *const c_void,
    // CHECK: ![[B]] = !DIDerivedType(tag: DW_TAG_member, name: "b", {{.*}}baseType: ![[B_BASE_TYPE:[0-9]+]]
    // CHECK: ![[B_BASE_TYPE]] = !DIDerivedType(tag: DW_TAG_pointer_type, name: "{{.*}}core::ffi::c_void{{.*}}", {{.*}}baseType: null
    b: *mut c_void,
}

pub fn aaa(_: &Foo) {}
