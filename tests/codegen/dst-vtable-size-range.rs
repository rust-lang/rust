//@ compile-flags: -Copt-level=3 -Z merge-functions=disabled

#![crate_type = "lib"]
#![feature(core_intrinsics)]

// Check that we annotate size loads from vtables with 0..(isize::MAX + 1) range metadata.

pub trait Trait {
    fn f(&self);
}

// Note that rustc uses inclusive bounds, but LLVM uses exclusive bounds for range metadata.
// CHECK-LABEL: @generate_exclusive_bound
#[no_mangle]
pub fn generate_exclusive_bound() -> usize {
    // CHECK: ret [[USIZE:i[0-9]+]] [[EXCLUSIVE_BOUND:[-0-9]+]]
    isize::MAX as usize + 1
}

// CHECK-LABEL: @size_load_from_size_of_val
#[no_mangle]
pub fn size_load_from_size_of_val(x: &dyn Trait) -> usize {
    // CHECK: {{%[0-9]+}} = load [[USIZE]], {{.+}} !range [[RANGE_META:![0-9]+]]
    core::mem::size_of_val(x)
}

// CHECK-LABEL: @size_load_from_vtable_size_intrinsic
#[no_mangle]
pub unsafe fn size_load_from_vtable_size_intrinsic(x: &dyn Trait) -> usize {
    let (data, vtable): (*const (), *const ()) = core::mem::transmute(x);
    // CHECK: {{%[0-9]+}} = load [[USIZE]], {{.+}} !range [[RANGE_META]]
    core::intrinsics::vtable_size(vtable)
}

// CHECK: [[RANGE_META]] = !{[[USIZE]] 0, [[USIZE]] [[EXCLUSIVE_BOUND]]}
