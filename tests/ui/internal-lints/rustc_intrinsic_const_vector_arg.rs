// compile-flags: -Z unstable-options
#![feature(abi_unadjusted)]
#![feature(inline_const)]
#![feature(intrinsics)]
#![allow(non_camel_case_types)]
#![feature(repr_simd)]
#![feature(rustc_attrs)]
#![feature(simd_ffi)]
#![allow(unused)]

#[repr(simd)]
#[derive(Clone)]
pub struct i8x2(i8, i8);

extern "unadjusted" {
    #[rustc_intrinsic_const_vector_arg] //~ ERROR malformed `rustc_intrinsic_const_vector_arg` attribute input
    fn foo1(a: i8x2, b: i8);
}

extern "unadjusted" {
    #[rustc_intrinsic_const_vector_arg = "1"] //~ ERROR malformed `rustc_intrinsic_const_vector_arg` attribute input
    fn foo2(a: i8x2, b: i8);
}

#[rustc_intrinsic_const_vector_arg(0)] //~ ERROR  attribute should be applied to functions in `extern "unadjusted"` modules
pub struct foo3(i8x2);

extern "C" {
    #[rustc_intrinsic_const_vector_arg(0)] //~ ERROR  attribute should be applied to functions in `extern "unadjusted"` modules
    fn foo4(a: i8x2);
}

extern "unadjusted" {
    #[rustc_intrinsic_const_vector_arg(0)] //~ ERROR function does not have a parameter at index 0
    fn foo5();
}

extern "unadjusted" {
    #[rustc_intrinsic_const_vector_arg(1)] //~ ERROR function does not have a parameter at index 1
    fn foo6(a: i8x2);
}

extern "unadjusted" {
    #[rustc_intrinsic_const_vector_arg("bar")] //~ ERROR attribute requires a parameter index
    fn foo7(a: i8x2);
}

extern "unadjusted" {
    #[rustc_intrinsic_const_vector_arg(0,2)] //~ ERROR function does not have a parameter at index 2
    fn foo8(a: i8x2, b: i8);
}

extern "unadjusted" {
    #[rustc_intrinsic_const_vector_arg(0)] //~ ERROR parameter at index 0 must be a simd type
    fn foo9(a: i8);
}

fn main() {}
