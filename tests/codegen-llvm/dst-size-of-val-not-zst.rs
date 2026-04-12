//@ compile-flags: -Copt-level=3 -Z merge-functions=disabled
//@ min-llvm-version: 21
//@ needs-deterministic-layouts

#![crate_type = "lib"]

// Regression test for #152788: `size_of_val(p) == 0` folds to `false` for
// DSTs with a non-zero prefix (nuw+nsw on offset+tail, assume on rounding).

pub struct Foo<T: ?Sized>(pub [u32; 3], pub T);

// CHECK-LABEL: @size_of_val_dyn_not_zero
#[no_mangle]
pub fn size_of_val_dyn_not_zero(p: &Foo<dyn std::fmt::Debug>) -> bool {
    // CHECK: ret i1 false
    std::mem::size_of_val(p) == 0
}

// CHECK-LABEL: @size_of_val_slice_u8_not_zero
#[no_mangle]
pub fn size_of_val_slice_u8_not_zero(p: &Foo<[u8]>) -> bool {
    // CHECK: ret i1 false
    std::mem::size_of_val(p) == 0
}

// CHECK-LABEL: @size_of_val_slice_i32_not_zero
#[no_mangle]
pub fn size_of_val_slice_i32_not_zero(p: &Foo<[i32]>) -> bool {
    // CHECK: ret i1 false
    std::mem::size_of_val(p) == 0
}
