// run-pass
#![allow(unused_must_use)]
// ignore-emscripten FIXME(#45351) hits an LLVM assert

#![feature(repr_simd, platform_intrinsics, concat_idents, test)]
#![allow(non_camel_case_types)]

extern crate test;

#[repr(simd)]
#[derive(PartialEq, Debug)]
struct i32x4(i32, i32, i32, i32);
#[repr(simd)]
#[derive(PartialEq, Debug)]
struct i8x4(i8, i8, i8, i8);

#[repr(simd)]
#[derive(PartialEq, Debug)]
struct u32x4(u32, u32, u32, u32);
#[repr(simd)]
#[derive(PartialEq, Debug)]
struct u8x4(u8, u8, u8, u8);

#[repr(simd)]
#[derive(PartialEq, Debug)]
struct f32x4(f32, f32, f32, f32);

#[repr(simd)]
#[derive(PartialEq, Debug)]
struct f64x4(f64, f64, f64, f64);


extern "platform-intrinsic" {
    fn simd_cast<T, U>(x: T) -> U;
}

const A: i32 = -1234567;
const B: i32 = 12345678;
const C: i32 = -123456789;
const D: i32 = 1234567890;

trait Foo {
    fn is_float() -> bool { false }
    fn in_range(x: i32) -> bool;
}
impl Foo for i32 {
    fn in_range(_: i32) -> bool { true }
}
impl Foo for i8 {
    fn in_range(x: i32) -> bool { -128 <= x && x < 128 }
}
impl Foo for u32 {
    fn in_range(x: i32) -> bool { 0 <= x }
}
impl Foo for u8 {
    fn in_range(x: i32) -> bool { 0 <= x && x < 128 }
}
impl Foo for f32 {
    fn is_float() -> bool { true }
    fn in_range(_: i32) -> bool { true }
}
impl Foo for f64 {
    fn is_float() -> bool { true }
    fn in_range(_: i32) -> bool { true }
}

fn main() {
    macro_rules! test {
        ($from: ident, $to: ident) => {{
            // force the casts to actually happen, or else LLVM/rustc
            // may fold them and get slightly different results.
            let (a, b, c, d) = test::black_box((A as $from, B as $from, C as $from, D as $from));
            // the SIMD vectors are all FOOx4, so we can concat_idents
            // so we don't have to pass in the extra args to the macro
            let mut from = simd_cast(concat_idents!($from, x4)(a, b, c, d));
            let mut to = concat_idents!($to, x4)(a as $to,
                                                 b as $to,
                                                 c as $to,
                                                 d as $to);
            // assist type inference, it needs to know what `from` is
            // for the `if` statements.
            to == from;

            // there are platform differences for some out of range
            // casts, so we just normalize such things: it's OK for
            // "invalid" calculations to result in nonsense answers.
            // (e.g., negative float to unsigned integer goes through a
            // library routine on the default i686 platforms, and the
            // implementation of that routine differs on e.g., Linux
            // vs. macOS, resulting in different answers.)
            if $from::is_float() {
                if !$to::in_range(A) { from.0 = 0 as $to; to.0 = 0 as $to; }
                if !$to::in_range(B) { from.1 = 0 as $to; to.1 = 0 as $to; }
                if !$to::in_range(C) { from.2 = 0 as $to; to.2 = 0 as $to; }
                if !$to::in_range(D) { from.3 = 0 as $to; to.3 = 0 as $to; }
            }

            assert!(to == from,
                    "{} -> {} ({:?} != {:?})", stringify!($from), stringify!($to),
                    from, to);
        }}
    }
    macro_rules! tests {
        (: $($to: ident),*) => { () };
        // repeating the list twice is easier than writing a cartesian
        // product macro
        ($from: ident $(, $from_: ident)*: $($to: ident),*) => {
            fn $from() { unsafe { $( test!($from, $to); )* } }
            tests!($($from_),*: $($to),*)
        };
        ($($types: ident),*) => {{
            tests!($($types),* : $($types),*);
            $($types();)*
        }}
    }

    // test various combinations, including truncation,
    // signed/unsigned extension, and floating point casts.
    tests!(i32, i8, u32, u8, f32);
    tests!(i32, u32, f32, f64)
}
