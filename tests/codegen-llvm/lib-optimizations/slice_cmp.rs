//@ compile-flags: -Copt-level=3
#![crate_type = "lib"]


// Comparing slices of pointers should be vectorizable.

#[derive(PartialEq)]
struct PtrWrapper(*const u8);


// CHECK-LABEL: @slice_cmp_ptr_newtype
#[no_mangle]
pub fn slice_cmp_ptr_newtype(a: &[PtrWrapper], b: &[PtrWrapper]) -> bool {
    // CHECK:  icmp ne <4 x ptr>
    // CHECK:  ret i1
    a == b
}


// CHECK-LABEL: @slice_cmp_ptr
#[no_mangle]
pub fn slice_cmp_ptr(a: &[*const u8], b: &[*const u8]) -> bool {
    // CHECK:  icmp ne <4 x ptr>
    // CHECK:  ret i1
    a == b
}
