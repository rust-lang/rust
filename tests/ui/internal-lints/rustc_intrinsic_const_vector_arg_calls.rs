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

#[repr(simd)]
#[derive(Clone)]
pub struct f32x2(f32, f32);

#[repr(simd)]
#[derive(Clone)]
pub struct i8x2_arr([i8; 2]);

#[repr(simd)]
#[derive(Clone)]
pub struct f32x2_arr([f32; 2]);

extern "unadjusted" {
    #[rustc_intrinsic_const_vector_arg(0)] // OK
    fn foo1(a: i8x2);
}

extern "unadjusted" {
    #[rustc_intrinsic_const_vector_arg(0, 1)] // OK
    fn foo2(a: i8x2, b: i8x2);
}

extern "unadjusted" {
    #[rustc_intrinsic_const_vector_arg(0)] // OK
    fn foo3(a: i8x2_arr);
}

extern "unadjusted" {
    #[rustc_intrinsic_const_vector_arg(0)] // OK
    fn foo4(a: f32x2);
}

extern "unadjusted" {
    #[rustc_intrinsic_const_vector_arg(0)] // OK
    fn foo5(a: f32x2_arr);
}

fn main() {
    unsafe {
        foo1(i8x2(0, 1)); //~ ERROR argument at index 0 must be a constant
        foo1({ i8x2(0, 1) }); //~ ERROR argument at index 0 must be a constant
        foo1(const { i8x2(0, 1) }); // OK

        foo2(const { i8x2(0, 1) }, { i8x2(2, 3) }); //~ ERROR argument at index 1 must be a constant
        foo2(const { i8x2(0, 1) }, const { i8x2(2, 3) }); // OK

        foo3(i8x2_arr([0, 1])); //~ ERROR argument at index 0 must be a constant
        foo3(const { i8x2_arr([0, 1]) }); // OK

        foo4(f32x2(0.0, 1.0)); //~ ERROR argument at index 0 must be a constant
        foo4(const { f32x2(0.0, 1.0) }); // OK

        foo5(f32x2_arr([0.0, 1.0])); //~ ERROR argument at index 0 must be a constant
        foo5(const { f32x2_arr([0.0, 1.0]) }); // OK
    }
}
