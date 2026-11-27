// Check that appropriate errors are reported if an intrinsic is defined
// with the wrong number of generic lifetime/type/const parameters, and
// that no ICE occurs in these cases.

#![feature(intrinsics)]
#![crate_type = "lib"]

#[rustc_intrinsic]
unsafe fn simd_saturating_add<'a, T: 'a>(x: T, y: T);
//~^ ERROR: intrinsic has wrong number of lifetime parameters

#[rustc_intrinsic]
unsafe fn simd_add<'a, T>(x: T, y: T) -> T;

#[rustc_intrinsic]
unsafe fn simd_sub<T, U>(x: T, y: U);
//~^ ERROR: intrinsic has wrong number of type parameters

#[rustc_intrinsic]
unsafe fn simd_mul<T, const N: usize>(x: T, y: T);
//~^ ERROR: intrinsic has wrong number of const parameters
