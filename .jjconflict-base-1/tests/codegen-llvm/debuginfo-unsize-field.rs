//@ compile-flags:-g -Copt-level=0 -C panic=abort

// Check that debug information for structs with embedded str and [u8] slices is distinct from
// structs with embedded u8

#![crate_type = "lib"]

// NOTE: regex for the CHECK directives,
// depending on the target, u8/usize are basic types or typedefs
//   linux:  !1 = !DIBasicType(name: "u8",
//   win:    !1 = !DIDerivedType(tag: .*, name: "u8",
// and references types are
//   linux:  name: "&debuginfo_unsize_field::Foo"
//   win:    name: "ref$<debuginfo_unsize_field::Foo>"

// CHECK: ![[U8:[0-9]+]] = !DI{{Basic|Derived}}Type({{.*}}name: "u8",

pub struct Foo {
    a: u32,
    b: str,
}
// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "{{&|ref\$<}}{{[^"]+}}::Foo{{>?}}", {{.*}}elements: ![[FOO_REF_ELEMS:[0-9]+]]
// CHECK: ![[FOO_REF_ELEMS]] = !{![[FOO_REF_PTR:[0-9]+]], ![[FOO_REF_LEN:[0-9]+]]}
// CHECK: ![[FOO_REF_PTR]] = !DIDerivedType(tag: DW_TAG_member, name: "data_ptr", {{.*}}baseType: ![[FOO_PTR:[0-9]+]]
// CHECK: ![[FOO_PTR]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[FOO:[0-9]+]]
// CHECK: ![[FOO]] = !DICompositeType(tag: DW_TAG_structure_type, name: "Foo", {{.*}}elements: ![[FOO_ELEMS:[0-9]+]]
// CHECK: ![[FOO_ELEMS]] = !{![[FOO_A:[0-9]+]], ![[FOO_B:[0-9]+]]}
// CHECK: ![[FOO_A]] = !DIDerivedType(tag: DW_TAG_member, name: "a"
// CHECK: ![[FOO_B]] = !DIDerivedType(tag: DW_TAG_member, name: "b", {{.*}}baseType: ![[U8_SLICE:[0-9]+]]
//
// CHECK: ![[U8_SLICE]] = !DICompositeType(tag: DW_TAG_array_type, baseType: ![[U8]], {{.*}}elements: ![[U8_SLICE_ELEMS:[0-9]+]]
// CHECK: ![[U8_SLICE_ELEMS]] = !{![[U8_SLICE_RANGE:[0-9]+]]}
// this is special to embedded slices, there is no upper bound on the number of elements,
// that info is stored in the length metadata for a reference to the parent struct
// CHECK: ![[U8_SLICE_RANGE]] = !DISubrange(count: -1, lowerBound: 0)
//
// CHECK: ![[FOO_REF_LEN]] = !DIDerivedType(tag: DW_TAG_member, name: "length", {{.*}}baseType: ![[USIZE:[0-9]+]]
// CHECK: ![[USIZE]] = !DI{{Basic|Derived}}Type({{.*}}name: "usize"
pub struct Bar {
    a: u32,
    b: [u8],
}
// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "{{&|ref\$<}}{{[^"]+}}::Bar{{>?}}", {{.*}}elements: ![[BAR_REF_ELEMS:[0-9]+]]
// CHECK: ![[BAR_REF_ELEMS]] = !{![[BAR_REF_PTR:[0-9]+]], ![[BAR_REF_LEN:[0-9]+]]}
// CHECK: ![[BAR_REF_PTR]] = !DIDerivedType(tag: DW_TAG_member, name: "data_ptr", {{.*}}baseType: ![[BAR_PTR:[0-9]+]]
// CHECK: ![[BAR_PTR]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[BAR:[0-9]+]]
// CHECK: ![[BAR]] = !DICompositeType(tag: DW_TAG_structure_type, name: "Bar", {{.*}}elements: ![[BAR_ELEMS:[0-9]+]]
// CHECK: ![[BAR_ELEMS]] = !{![[BAR_A:[0-9]+]], ![[BAR_B:[0-9]+]]}
// CHECK: ![[BAR_A]] = !DIDerivedType(tag: DW_TAG_member, name: "a"
// CHECK: ![[BAR_B]] = !DIDerivedType(tag: DW_TAG_member, name: "b", {{.*}}baseType: ![[U8_SLICE]]
// CHECK: ![[BAR_REF_LEN]] = !DIDerivedType(tag: DW_TAG_member, name: "length", {{.*}}baseType: ![[USIZE:[0-9]+]]
pub struct Baz {
    a: u32,
    b: u8,
}
// CHECK: !DIDerivedType(tag: DW_TAG_pointer_type, name: "{{&|ref\$<}}{{[^"]+}}::Baz{{>?}}", {{.*}}baseType: ![[BAZ:[0-9]+]]
// CHECK: ![[BAZ]] = !DICompositeType(tag: DW_TAG_structure_type, name: "Baz", {{.*}}elements: ![[BAZ_ELEMS:[0-9]+]]
// CHECK: ![[BAZ_ELEMS]] = !{![[BAZ_A:[0-9]+]], ![[BAZ_B:[0-9]+]]}
// CHECK: ![[BAZ_A]] = !DIDerivedType(tag: DW_TAG_member, name: "a"
// CHECK: ![[BAZ_B]] = !DIDerivedType(tag: DW_TAG_member, name: "b", {{.*}}baseType: ![[U8]]

#[no_mangle]
pub fn test<'a>(a: &'a Foo, b: &'a Bar, c: &'a Baz) -> &'a u8 {
    // just use this somehow so the debuginfo isn't removed
    &a.b.as_bytes()[0]
}
