//@ compile-flags: -Zautodiff=Enable -Zautodiff=NoPostopt -C opt-level=3  -Clto=fat
//@ no-prefer-dynamic
//@ needs-enzyme
//
#![feature(rustc_attrs)]
#![feature(prelude_import)]
#![feature(panic_internals)]
#![no_std]
//@ needs-enzyme
#![feature(autodiff)]
#[prelude_import]
use ::std::prelude::rust_2015::*;
#[macro_use]
extern crate std;
//@ pretty-mode:expanded
//@ pretty-compare-only
//@ pp-exact:batching.pp


// Test that forward mode ad macros are expanded correctly.
use std::arch::asm;
use std::autodiff::autodiff;

// CHECK: ; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind nonlazybind willreturn memory(argmem: read) uwtable
// CHECK-NEXT: define dso_local noundef float @square(ptr noalias nocapture noundef readonly align 4 dereferenceable(4) %x) unnamed_addr #2 {
// CHECK-NEXT: start:
// CHECK-NEXT:   %_2 = load float, ptr %x, align 4, !noundef !4
// CHECK-NEXT:   %_0 = fmul float %_2, %_2
// CHECK-NEXT:   ret float %_0
// CHECK-NEXT: }

// CHECK: ; Function Attrs: alwaysinline nonlazybind uwtable
// CHECK-NEXT: define dso_local void @d_square2(ptr dead_on_unwind noalias nocapture noundef writable writeonly sret([16 x i8]) align 4 dereferenceable(16) initializes((0, 16)) %_0, ptr noalias nocapture noundef readonly align 4 dereferenceable(4) %x, ptr noalias noundef readonly align 4 dereferenceable(4) %bx_0, ptr noalias noundef readonly align 4 dereferenceable(4) %bx_1, ptr noalias noundef readonly align 4 dereferenceable(4) %bx_2) unnamed_addr #3 personality ptr @rust_eh_personality {
// CHECK-NEXT: start:
// CHECK-NEXT:   %0 = insertvalue [4 x ptr] undef, ptr %x, 0
// CHECK-NEXT:   %1 = insertvalue [4 x ptr] %0, ptr %bx_0, 1
// CHECK-NEXT:   %2 = insertvalue [4 x ptr] %1, ptr %bx_1, 2
// CHECK-NEXT:   %3 = insertvalue [4 x ptr] %2, ptr %bx_2, 3
// CHECK-NEXT:   %4 = call [4 x float] @batch_square([4 x ptr] %3)
// CHECK-NEXT:   %.elt = extractvalue [4 x float] %4, 0
// CHECK-NEXT:   store float %.elt, ptr %_0, align 4
// CHECK-NEXT:   %_0.repack1 = getelementptr inbounds nuw i8, ptr %_0, i64 4
// CHECK-NEXT:   %.elt2 = extractvalue [4 x float] %4, 1
// CHECK-NEXT:   store float %.elt2, ptr %_0.repack1, align 4
// CHECK-NEXT:   %_0.repack3 = getelementptr inbounds nuw i8, ptr %_0, i64 8
// CHECK-NEXT:   %.elt4 = extractvalue [4 x float] %4, 2
// CHECK-NEXT:   store float %.elt4, ptr %_0.repack3, align 4
// CHECK-NEXT:   %_0.repack5 = getelementptr inbounds nuw i8, ptr %_0, i64 12
// CHECK-NEXT:   %.elt6 = extractvalue [4 x float] %4, 3
// CHECK-NEXT:   store float %.elt6, ptr %_0.repack5, align 4
// CHECK-NEXT:   ret void
// CHECK-NEXT: }

/// Generated from:
/// ```
/// #[batching(d_square2, 4, Vector, Vector)]
/// fn square(x: &f32) -> f32 {
///   x * x
/// }

#[no_mangle]
#[rustc_autodiff]
#[inline(never)]
fn square(x: &f32) -> f32 {
    x * x
}
#[rustc_autodiff(Batch, 4, Vector, Vector)]
#[no_mangle]
#[inline(never)]
fn d_square2(x: &f32, bx_0: &f32, bx_1: &f32, bx_2: &f32) -> [f32; 4usize] {
    unsafe {
        asm!("NOP", options(nomem));
    };
    ::core::hint::black_box(square(x));
    ::core::hint::black_box((bx_0, bx_1, bx_2));
    ::core::hint::black_box(<[f32; 4usize]>::default())
}


fn main() {}
