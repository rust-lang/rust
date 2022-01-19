// compile-flags: -Cdebuginfo=2
#![crate_type = "lib"]

// The program below results in debuginfo being generated for at least
// five different pointer types. Check that they don't have a
// dwarfAddressSpace attribute.

// CHECK: !DIDerivedType(tag: DW_TAG_pointer_type
// CHECK-NOT: dwarfAddressSpace

// CHECK: !DIDerivedType(tag: DW_TAG_pointer_type
// CHECK-NOT: dwarfAddressSpace

// CHECK: !DIDerivedType(tag: DW_TAG_pointer_type
// CHECK-NOT: dwarfAddressSpace

// CHECK: !DIDerivedType(tag: DW_TAG_pointer_type
// CHECK-NOT: dwarfAddressSpace

// CHECK: !DIDerivedType(tag: DW_TAG_pointer_type
// CHECK-NOT: dwarfAddressSpace

pub fn foo(
    shared_ref: &usize,
    mut_ref: &mut usize,
    const_ptr: *const usize,
    mut_ptr: *mut usize,
    boxed: Box<usize>,
) -> usize {
    *shared_ref + *mut_ref + const_ptr as usize + mut_ptr as usize + *boxed
}
