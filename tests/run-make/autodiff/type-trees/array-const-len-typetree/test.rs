#![crate_type = "lib"]
#![feature(autodiff)]

use std::autodiff::autodiff_reverse;

// Regression for #160635: anon-const array lengths in struct fields need
// normalization before typetree walks. `*mut [f32; N]` used to ICE in
// `struct_tail_for_codegen` (deepest trailing field / unsizing tail), and a
// plain `[f32; N]` field used to emit an empty TypeTree.
//
// `scale` is `i32` (not `f32`) so a naive `[-1]:Float` over the whole struct
// would misclassify it. Array metadata has to stay bounded to `data`.

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
    pub scale: i32,
}

#[no_mangle]
#[inline(never)]
pub unsafe fn copy_inline_array(a: &InlineArray, b: &mut InlineArray) {
    *b = *a;
}
