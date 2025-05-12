//@compile-flags: -Copt-level=3

#![crate_type = "lib"]
#![feature(repr_simd, core_intrinsics)]

#[allow(non_camel_case_types)]
#[derive(Clone, Copy)]
#[repr(simd)]
struct i32x4([i32; 4]);

#[inline(always)]
fn to_array4(a: i32x4) -> [i32; 4] {
    a.0
}

// CHECK-LABEL: simd_add_self_then_return_array(
// CHECK-SAME: ptr{{.+}}sret{{.+}}%[[RET:.+]],
// CHECK-SAME: ptr{{.+}}%a)
#[no_mangle]
pub fn simd_add_self_then_return_array(a: &i32x4) -> [i32; 4] {
    // It would be nice to just ban `.0` into simd types,
    // but until we do this has to keep working.
    // See also <https://github.com/rust-lang/rust/issues/105439>

    // CHECK: %[[T1:.+]] = load <4 x i32>, ptr %a
    // CHECK: %[[T2:.+]] = shl <4 x i32> %[[T1]], {{splat \(i32 1\)|<i32 1, i32 1, i32 1, i32 1>}}
    // CHECK: store <4 x i32> %[[T2]], ptr %[[RET]]
    let a = *a;
    let b = unsafe { core::intrinsics::simd::simd_add(a, a) };
    to_array4(b)
}
