// Check that appropriate errors are reported if an intrinsic is defined
// with the wrong number of generic lifetime/type/const parameters, and
// that no ICE occurs in these cases.

#![feature(platform_intrinsics)]
#![crate_type="lib"]

extern "platform-intrinsic" {
    fn simd_saturating_add<'a, T: 'a>(x: T, y: T);
    //~^ ERROR: intrinsic has wrong number of lifetime parameters

    fn simd_add<'a, T>(x: T, y: T) -> T;

    fn simd_sub<T, U>(x: T, y: U);
    //~^ ERROR: intrinsic has wrong number of type parameters

    fn simd_mul<T, const N: usize>(x: T, y: T);
    //~^ ERROR: intrinsic has wrong number of const parameters
}
