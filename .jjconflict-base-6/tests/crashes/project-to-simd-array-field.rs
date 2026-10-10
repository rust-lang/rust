//@ known-bug: #137108
//@compile-flags: -Copt-level=3

// If you fix this, put it in the corresponding codegen test,
// not in a UI test like the readme says.

#![crate_type = "lib"]

#![feature(repr_simd, core_intrinsics)]

#[allow(non_camel_case_types)]
#[derive(Clone, Copy)]
#[repr(simd)]
struct i32x3([i32; 3]);

const _: () = { assert!(size_of::<i32x3>() == 16) };

#[inline(always)]
fn to_array3(a: i32x3) -> [i32; 3] {
    a.0
}

// CHECK-LABEL: simd_add_self_then_return_array_packed(
// CHECK-SAME: ptr{{.+}}sret{{.+}}%[[RET:.+]],
// CHECK-SAME: ptr{{.+}}%a)
#[no_mangle]
pub fn simd_add_self_then_return_array_packed(a: i32x3) -> [i32; 3] {
    // CHECK: %[[T1:.+]] = load <3 x i32>, ptr %a
    // CHECK: %[[T2:.+]] = shl <3 x i32> %[[T1]], <i32 1, i32 1, i32 1>
    // CHECK: store <3 x i32> %[[T2]], ptr %[[RET]]
    let b = unsafe { core::intrinsics::simd::simd_add(a, a) };
    to_array3(b)
}
