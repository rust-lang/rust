#![feature(repr_simd)]
#![allow(non_camel_case_types)]

#[repr(simd)]
struct empty; //~ ERROR SIMD vector cannot be empty

#[repr(simd)]
struct empty2([f32; 0]); //~ ERROR SIMD vector cannot be empty

#[repr(simd)]
struct pow2([f32; 7]);

#[repr(simd)]
struct i64f64(i64, f64); //~ ERROR SIMD vector's only field must be an array

struct Foo;

#[repr(simd)]
struct FooV(Foo, Foo); //~ ERROR SIMD vector's only field must be an array

#[repr(simd)]
struct FooV2([Foo; 2]); //~ ERROR SIMD vector element type should be a primitive scalar (integer/float/pointer) type

#[repr(simd)]
struct TooBig([f32; 65536]); //~ ERROR SIMD vector cannot have more than 32768 elements

#[repr(simd)]
struct JustRight([u128; 32768]);

#[repr(simd)]
struct RGBA { //~ ERROR SIMD vector's only field must be an array
    r: f32,
    g: f32,
    b: f32,
    a: f32,
}

fn main() {}
