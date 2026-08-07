#![crate_type = "lib"]
#![feature(autodiff)]

use std::autodiff::autodiff_reverse;

// Regression test for #160635: non-literal array lengths (anon consts) in struct
// fields must be normalized before typetree recursion. Without that, `*mut [f32; N]`
// ICEs in `struct_tail_for_codegen`, and a plain `[f32; N]` field silently yields
// an empty TypeTree.

const N: usize = 8;

#[derive(Copy, Clone)]
#[repr(C)]
pub struct PtrArray {
    pub p: *mut [f32; N],
    pub q: f32,
    pub r: f32,
}

#[no_mangle]
#[inline(never)]
pub unsafe fn copy_ptr_array(a: &PtrArray, b: &mut PtrArray) {
    *b = *a;
}

// Run Enzyme over the ICE-shaped type so metadata is not only emitted but accepted.
#[autodiff_reverse(d_ptr_array_sum, Duplicated, Active)]
#[no_mangle]
#[inline(never)]
pub fn ptr_array_sum(s: &PtrArray) -> f32 {
    s.q + s.r
}

#[no_mangle]
pub fn exercise_ptr_array_sum(s: &PtrArray, ds: &mut PtrArray) -> f32 {
    d_ptr_array_sum(s, ds, 1.0)
}

#[derive(Copy, Clone)]
#[repr(C)]
pub struct InlineArray {
    pub data: [f32; N],
    pub scale: f32,
}

#[no_mangle]
#[inline(never)]
pub unsafe fn copy_inline_array(a: &InlineArray, b: &mut InlineArray) {
    *b = *a;
}
