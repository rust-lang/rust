// ignore-tidy-file-linelength
#![feature(complex_numbers, f128)]
#![allow(improper_ctypes)]
#![allow(unused_features)]
#![deny(dead_code)]

use std::ffi::*;
use std::num::Complex;

fn main() {
    sqrt();
    #[cfg(not(target_family = "wasm"))]
    mul();
    #[cfg(not(target_family = "wasm"))]
    div();
    pass_simple();
    aligned_int();
    aligned_float();
    spill_gpr();
    spill_fpr();
    partial_gpr();
}

// This is just a placeholder until we actually add c_longdouble.
#[allow(non_camel_case_types)]
#[allow(unused)]
type c_longdouble = cfg_select! {
    target_arch = "aarch64" => f128,
    _ => (),
};

fn sqrt() {
    unsafe extern "C" {
        safe fn csqrtf(_: Complex<c_float>) -> Complex<c_float>;
        safe fn csqrt(_: Complex<c_double>) -> Complex<c_double>;
        #[cfg(target_arch = "aarch64")]
        safe fn csqrtl(_: Complex<c_longdouble>) -> Complex<c_longdouble>;
    }

    let c = Complex::new(-1.0, 0.0);
    assert_eq!(csqrtf(c), Complex::new(0.0, 1.0));

    let c = Complex::new(-1.0, 0.0);
    assert_eq!(csqrt(c), Complex::new(0.0, 1.0));

    #[cfg(target_arch = "aarch64")]
    {
        let c = Complex::new(-1.0, 0.0);
        assert_eq!(csqrtl(c), Complex::new(0.0, 1.0));
    }
}

#[cfg(not(target_family = "wasm"))]
fn mul() {
    unsafe extern "C" {
        safe fn __mulsc3(a: c_float, b: c_float, c: c_float, d: c_float) -> Complex<c_float>;
        safe fn __muldc3(a: c_double, b: c_double, c: c_double, d: c_double) -> Complex<c_double>;
        #[cfg(target_arch = "aarch64")]
        safe fn __multc3(
            a: c_longdouble,
            b: c_longdouble,
            c: c_longdouble,
            d: c_longdouble,
        ) -> Complex<c_longdouble>;
    }

    assert_eq!(__mulsc3(1.0, 2.0, 3.0, 4.0), Complex::new(-5.0, 10.0));
    assert_eq!(__muldc3(1.0, 2.0, 3.0, 4.0), Complex::new(-5.0, 10.0));

    #[cfg(target_arch = "aarch64")]
    assert_eq!(__multc3(1.0, 2.0, 3.0, 4.0), Complex::new(-5.0, 10.0));

    // The naive algorithm would return NaN + NaNi for these inputs, but the libcall handles it.
    assert_eq!(
        __mulsc3(1.0, 0.0, c_float::INFINITY, c_float::INFINITY),
        Complex::new(c_float::INFINITY, c_float::INFINITY)
    );
    assert_eq!(
        __muldc3(1.0, 0.0, c_double::INFINITY, c_double::INFINITY),
        Complex::new(c_double::INFINITY, c_double::INFINITY)
    );

    #[cfg(target_arch = "aarch64")]
    assert_eq!(
        __multc3(1.0, 0.0, c_longdouble::INFINITY, c_longdouble::INFINITY),
        Complex::new(c_longdouble::INFINITY, c_longdouble::INFINITY)
    );
}

#[cfg(not(target_family = "wasm"))]
fn div() {
    unsafe extern "C" {
        safe fn __divsc3(a: c_float, b: c_float, c: c_float, d: c_float) -> Complex<c_float>;
        safe fn __divdc3(a: c_double, b: c_double, c: c_double, d: c_double) -> Complex<c_double>;
        #[cfg(target_arch = "aarch64")]
        safe fn __divtc3(
            a: c_longdouble,
            b: c_longdouble,
            c: c_longdouble,
            d: c_longdouble,
        ) -> Complex<c_longdouble>;
    }

    assert_eq!(__divsc3(2.0, 11.0, 2.0, 1.0), Complex::new(3.0, 4.0));
    assert_eq!(__divdc3(2.0, 11.0, 2.0, 1.0), Complex::new(3.0, 4.0));

    #[cfg(target_arch = "aarch64")]
    assert_eq!(__divtc3(2.0, 11.0, 2.0, 1.0), Complex::new(3.0, 4.0));

    assert_eq!(
        __divsc3(c_float::INFINITY, 0.0, 1.0, 1.0),
        Complex::new(c_float::INFINITY, c_float::NEG_INFINITY)
    );
    assert_eq!(
        __divdc3(c_double::INFINITY, 0.0, 1.0, 1.0),
        Complex::new(c_double::INFINITY, c_double::NEG_INFINITY)
    );

    #[cfg(target_arch = "aarch64")]
    assert_eq!(
        __divtc3(c_longdouble::INFINITY, 0.0, 1.0, 1.0),
        Complex::new(c_longdouble::INFINITY, c_longdouble::NEG_INFINITY)
    );
}

fn pass_simple() {
    #[rustfmt::skip]
    #[link(name = "test", kind = "static")]
    unsafe extern "C" {
        safe fn pass_simple_complex_float(x: Complex<c_float>) -> Complex<c_float>;
        safe fn pass_simple_complex_double(x: Complex<c_double>) -> Complex<c_double>;
        #[cfg(target_arch = "aarch64")]
        safe fn pass_simple_complex_long_double(x: Complex<c_longdouble>) -> Complex<c_longdouble>;

        safe fn pass_simple_complex_char(x: Complex<c_char>) -> Complex<c_char>;
        safe fn pass_simple_complex_short(x: Complex<c_short>) -> Complex<c_short>;
        safe fn pass_simple_complex_int(x: Complex<c_int>) -> Complex<c_int>;
        safe fn pass_simple_complex_long(x: Complex<c_long>) -> Complex<c_long>;
        safe fn pass_simple_complex_long_long(x: Complex<c_longlong>) -> Complex<c_longlong>;
    }

    assert_eq!(Complex::new(3.5, 4.5), pass_simple_complex_float(Complex::new(3.5, 4.5)));
    assert_eq!(Complex::new(3.5, 4.5), pass_simple_complex_double(Complex::new(3.5, 4.5)));

    assert_eq!(
        Complex::new(c_float::INFINITY, c_float::NEG_INFINITY),
        pass_simple_complex_float(Complex::new(c_float::INFINITY, c_float::NEG_INFINITY))
    );
    assert_eq!(
        Complex::new(c_double::INFINITY, c_double::NEG_INFINITY),
        pass_simple_complex_double(Complex::new(c_double::INFINITY, c_double::NEG_INFINITY))
    );
    #[cfg(target_arch = "aarch64")]
    assert_eq!(
        Complex::new(c_longdouble::INFINITY, c_longdouble::NEG_INFINITY),
        pass_simple_complex_long_double(Complex::new(
            c_longdouble::INFINITY,
            c_longdouble::NEG_INFINITY
        ))
    );

    assert_eq!(Complex::new(3, 4), pass_simple_complex_char(Complex::new(3, 4)));
    assert_eq!(Complex::new(3, 4), pass_simple_complex_short(Complex::new(3, 4)));
    assert_eq!(Complex::new(3, 4), pass_simple_complex_int(Complex::new(3, 4)));
    assert_eq!(Complex::new(3, 4), pass_simple_complex_long(Complex::new(3, 4)));
    assert_eq!(Complex::new(3, 4), pass_simple_complex_long_long(Complex::new(3, 4)));

    assert_eq!(
        Complex::new(c_char::MIN, c_char::MAX),
        pass_simple_complex_char(Complex::new(c_char::MIN, c_char::MAX))
    );
    assert_eq!(
        Complex::new(c_short::MIN, c_short::MAX),
        pass_simple_complex_short(Complex::new(c_short::MIN, c_short::MAX))
    );
    assert_eq!(
        Complex::new(c_int::MIN, c_int::MAX),
        pass_simple_complex_int(Complex::new(c_int::MIN, c_int::MAX))
    );
    assert_eq!(
        Complex::new(c_long::MIN, c_long::MAX),
        pass_simple_complex_long(Complex::new(c_long::MIN, c_long::MAX))
    );
    assert_eq!(
        Complex::new(c_longlong::MIN, c_longlong::MAX),
        pass_simple_complex_long_long(Complex::new(c_longlong::MIN, c_longlong::MAX))
    );
}

fn aligned_int() {
    #[rustfmt::skip]
    #[link(name = "test", kind = "static")]
    unsafe extern "C" {
        safe fn complex_float_align_int(_: c_int, x: Complex<c_float>) -> Complex<c_float>;
        safe fn complex_double_align_int(_: c_int, x: Complex<c_double>) -> Complex<c_double>;
        #[cfg(target_arch = "aarch64")]
        safe fn complex_long_double_align_int(_: c_int, x: Complex<c_longdouble>) -> Complex<c_longdouble>;

        safe fn complex_char_align_int(_: c_int, x: Complex<c_char>) -> Complex<c_char>;
        safe fn complex_short_align_int(_: c_int, x: Complex<c_short>) -> Complex<c_short>;
        safe fn complex_int_align_int(_: c_int, x: Complex<c_int>) -> Complex<c_int>;
        safe fn complex_long_align_int(_: c_int, x: Complex<c_long>) -> Complex<c_long>;
        safe fn complex_long_long_align_int(_: c_int, x: Complex<c_longlong>) -> Complex<c_longlong>;
    }

    let a = 0xAAAA_AAAAu32 as c_int;

    assert_eq!(Complex::new(3.5, 4.5), complex_float_align_int(a, Complex::new(3.5, 4.5)));
    assert_eq!(Complex::new(3.5, 4.5), complex_double_align_int(a, Complex::new(3.5, 4.5)));

    assert_eq!(
        Complex::new(c_float::INFINITY, c_float::NEG_INFINITY),
        complex_float_align_int(a, Complex::new(c_float::INFINITY, c_float::NEG_INFINITY))
    );
    assert_eq!(
        Complex::new(c_double::INFINITY, c_double::NEG_INFINITY),
        complex_double_align_int(a, Complex::new(c_double::INFINITY, c_double::NEG_INFINITY))
    );
    #[cfg(target_arch = "aarch64")]
    assert_eq!(
        Complex::new(c_longdouble::INFINITY, c_longdouble::NEG_INFINITY),
        complex_long_double_align_int(
            a,
            Complex::new(c_longdouble::INFINITY, c_longdouble::NEG_INFINITY)
        )
    );

    assert_eq!(Complex::new(3, 4), complex_char_align_int(a, Complex::new(3, 4)));
    assert_eq!(Complex::new(3, 4), complex_short_align_int(a, Complex::new(3, 4)));
    assert_eq!(Complex::new(3, 4), complex_int_align_int(a, Complex::new(3, 4)));
    assert_eq!(Complex::new(3, 4), complex_long_align_int(a, Complex::new(3, 4)));
    assert_eq!(Complex::new(3, 4), complex_long_long_align_int(a, Complex::new(3, 4)));

    assert_eq!(
        Complex::new(c_char::MIN, c_char::MAX),
        complex_char_align_int(a, Complex::new(c_char::MIN, c_char::MAX))
    );
    assert_eq!(
        Complex::new(c_short::MIN, c_short::MAX),
        complex_short_align_int(a, Complex::new(c_short::MIN, c_short::MAX))
    );
    assert_eq!(
        Complex::new(c_int::MIN, c_int::MAX),
        complex_int_align_int(a, Complex::new(c_int::MIN, c_int::MAX))
    );
    assert_eq!(
        Complex::new(c_long::MIN, c_long::MAX),
        complex_long_align_int(a, Complex::new(c_long::MIN, c_long::MAX))
    );
    assert_eq!(
        Complex::new(c_longlong::MIN, c_longlong::MAX),
        complex_long_long_align_int(a, Complex::new(c_longlong::MIN, c_longlong::MAX))
    );
}

fn aligned_float() {
    #[rustfmt::skip]
    #[link(name = "test", kind = "static")]
    unsafe extern "C" {
        safe fn complex_float_align_float(_: c_float, x: Complex<c_float>) -> Complex<c_float>;
        safe fn complex_double_align_float(_: c_float, x: Complex<c_double>) -> Complex<c_double>;
    #[cfg(target_arch = "aarch64")]
        safe fn complex_long_double_align_float(_: c_float, x: Complex<c_longdouble>) -> Complex<c_longdouble>;

        safe fn complex_char_align_float(_: c_float, x: Complex<c_char>) -> Complex<c_char>;
        safe fn complex_short_align_float(_: c_float, x: Complex<c_short>) -> Complex<c_short>;
        safe fn complex_int_align_float(_: c_float, x: Complex<c_int>) -> Complex<c_int>;
        safe fn complex_long_align_float(_: c_float, x: Complex<c_long>) -> Complex<c_long>;
        safe fn complex_long_long_align_float(_: c_float, x: Complex<c_longlong>) -> Complex<c_longlong>;
    }

    let a = 3.14 as c_float;

    assert_eq!(Complex::new(3.5, 4.5), complex_float_align_float(a, Complex::new(3.5, 4.5)));
    assert_eq!(Complex::new(3.5, 4.5), complex_double_align_float(a, Complex::new(3.5, 4.5)));

    assert_eq!(
        Complex::new(c_float::INFINITY, c_float::NEG_INFINITY),
        complex_float_align_float(a, Complex::new(c_float::INFINITY, c_float::NEG_INFINITY))
    );
    assert_eq!(
        Complex::new(c_double::INFINITY, c_double::NEG_INFINITY),
        complex_double_align_float(a, Complex::new(c_double::INFINITY, c_double::NEG_INFINITY))
    );
    #[cfg(target_arch = "aarch64")]
    assert_eq!(
        Complex::new(c_longdouble::INFINITY, c_longdouble::NEG_INFINITY),
        complex_long_double_align_float(
            a,
            Complex::new(c_longdouble::INFINITY, c_longdouble::NEG_INFINITY)
        )
    );

    assert_eq!(Complex::new(3, 4), complex_char_align_float(a, Complex::new(3, 4)));
    assert_eq!(Complex::new(3, 4), complex_short_align_float(a, Complex::new(3, 4)));
    assert_eq!(Complex::new(3, 4), complex_int_align_float(a, Complex::new(3, 4)));
    assert_eq!(Complex::new(3, 4), complex_long_align_float(a, Complex::new(3, 4)));
    assert_eq!(Complex::new(3, 4), complex_long_long_align_float(a, Complex::new(3, 4)));

    assert_eq!(
        Complex::new(c_char::MIN, c_char::MAX),
        complex_char_align_float(a, Complex::new(c_char::MIN, c_char::MAX))
    );
    assert_eq!(
        Complex::new(c_short::MIN, c_short::MAX),
        complex_short_align_float(a, Complex::new(c_short::MIN, c_short::MAX))
    );
    assert_eq!(
        Complex::new(c_int::MIN, c_int::MAX),
        complex_int_align_float(a, Complex::new(c_int::MIN, c_int::MAX))
    );
    assert_eq!(
        Complex::new(c_long::MIN, c_long::MAX),
        complex_long_align_float(a, Complex::new(c_long::MIN, c_long::MAX))
    );
    assert_eq!(
        Complex::new(c_longlong::MIN, c_longlong::MAX),
        complex_long_long_align_float(a, Complex::new(c_longlong::MIN, c_longlong::MAX))
    );
}

#[rustfmt::skip]
fn spill_gpr() {
    #[link(name = "test", kind = "static")]
    unsafe extern "C" {
        fn spill_trailing_complex_float_1(a: c_int, b: c_int, c: c_int, d: c_int, e: c_int, f: c_int, g: c_int, x: Complex<c_float>, h: c_int) -> c_int;

        fn spill_trailing_complex_double_1(a: c_int, b: c_int, c: c_int, d: c_int, e: c_int, f: c_int, x: Complex<c_double>, g: c_int) -> c_int;
        fn spill_trailing_complex_double_2(a: c_int, b: c_int, c: c_int, d: c_int, e: c_int, x: Complex<c_double>, f: c_int, g: c_int) -> c_int;
        fn spill_trailing_complex_double_3(a: c_int, b: c_int, c: c_int, d: c_int, x: Complex<c_double>, e: c_int, f: c_int, g: c_int) -> c_int;

        #[cfg(target_arch = "aarch64")]
        fn spill_trailing_complex_long_double_1(a: c_int, x: Complex<c_longdouble>, b: c_int) -> c_int;
    }

    unsafe {
        assert_eq!(
            spill_trailing_complex_float_1(0x11111111, 0x22222222, 0x33333333, 0x44444444, 0x55555555, 0x66666666, 0x77777777, Complex::new(1.0, 2.0), 0x88888888_u32 as c_int),
            0x88888888_u32 as c_int,
        );

        assert_eq!(
            spill_trailing_complex_double_1(0x11111111, 0x22222222, 0x33333333, 0x44444444, 0x55555555, 0x66666666, Complex::new(1.0, 2.0), 0x77777777),
            0x77777777,
        );

        assert_eq!(
            spill_trailing_complex_double_2(0x11111111, 0x22222222, 0x33333333, 0x44444444, 0x55555555, Complex::new(1.0, 2.0), 0x66666666, 0x77777777),
            0x77777777,
        );

        assert_eq!(
            spill_trailing_complex_double_3(0x11111111, 0x22222222, 0x33333333, 0x44444444, Complex::new(1.0, 2.0), 0x55555555, 0x66666666, 0x77777777),
            0x77777777,
        );

        #[cfg(target_arch = "aarch64")]
        assert_eq!(
            spill_trailing_complex_long_double_1(0x11111111, Complex::new(1.0, 2.0), 0x22222222,),
            0x22222222,
        );
    }
}

#[rustfmt::skip]
fn spill_fpr() {
    #[link(name = "test", kind = "static")]
    unsafe extern "C" {
        safe fn spill_trailing_complex_float_1_float(a0: f32, a1: f32, a2: f32, a3: f32, a4: f32, a5: f32, a6: f32, value: Complex<f32>, x: f32) -> f32;
        safe fn spill_trailing_complex_float_2_float(a0: f32, a1: f32, a2: f32, a3: f32, a4: f32, a5: f32, a6: f32, a7: f32, value: Complex<f32>, x: f32) -> f32;

        safe fn spill_trailing_complex_double_1_float(a0: f32, a1: f32, a2: f32, a3: f32, a4: f32, a5: f32, value: Complex<f64>, x: f32) -> f32;
        safe fn spill_trailing_complex_double_2_float(a0: f32, a1: f32, a2: f32, a3: f32, a4: f32, a5: f32, a6: f32, value: Complex<f64>, x: f32, y: f32) -> f32;
        safe fn spill_trailing_complex_double_3_float(a0: f32, a1: f32, a2: f32, a3: f32, a4: f32, a5: f32, a6: f32, a7: f32, value: Complex<f64>, x: f32, y: f32) -> f32;

        #[cfg(target_arch = "aarch64")]
        safe fn spill_trailing_complex_long_double_1_float(a0: f32, a1: f32, a2: f32, a3: f32, a4: f32, a5: f32, value: Complex<c_longdouble>, x: f32) -> f32;
    }

    assert_eq!(
        spill_trailing_complex_float_1_float(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, Complex::new(1.0, 2.0), 0.8),
        0.8,
    );
    assert_eq!(
       spill_trailing_complex_float_2_float(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, Complex::new(1.0, 2.0), 0.8),
        0.8,
    );

    assert_eq!(
        spill_trailing_complex_double_1_float(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, Complex::new(1.0, 2.0), 0.6),
        0.6,
    );
    assert_eq!(
        spill_trailing_complex_double_2_float(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, Complex::new(1.0, 2.0), 0.7, 0.8),
        0.8,
    );
    assert_eq!(
        spill_trailing_complex_double_3_float(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, Complex::new(1.0, 2.0), 0.8, 0.9),
        0.9,
    );

    #[cfg(target_arch = "aarch64")]
    assert_eq!(
        spill_trailing_complex_long_double_1_float(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, Complex::new(1.0, 2.0), 0.6),
        0.6,
    );
}

#[rustfmt::skip]
fn partial_gpr() {
    #[link(name = "test", kind = "static")]
    unsafe extern "C" {
        safe fn partial_complex_float(a: c_int, b: c_int, c: c_int, d: c_int, e: c_int, f: c_int, g: c_int, x: Complex<c_float>) -> Complex<c_float>;
        safe fn partial_complex_double(a: c_int, b: c_int, c: c_int, d: c_int, e: c_int, f: c_int, g: c_int, x: Complex<c_double>) -> Complex<c_double>;
        #[cfg(target_arch = "aarch64")]
        safe fn partial_complex_long_double(a: c_int, b: c_int, c: c_int, d: c_int, e: c_int, f: c_int, g: c_int, x: Complex<c_longdouble>) -> Complex<c_longdouble>;

        safe fn partial_complex_char(a: c_int, b: c_int, c: c_int, d: c_int, e: c_int, f: c_int, g: c_int, x: Complex<c_schar>) -> Complex<c_schar>;
        safe fn partial_complex_short(a: c_int, b: c_int, c: c_int, d: c_int, e: c_int, f: c_int, g: c_int, x: Complex<c_short>) -> Complex<c_short>;
        safe fn partial_complex_int(a: c_int, b: c_int, c: c_int, d: c_int, e: c_int, f: c_int, g: c_int, x: Complex<c_int>) -> Complex<c_int>;
        safe fn partial_complex_long(a: c_int, b: c_int, c: c_int, d: c_int, e: c_int, f: c_int, g: c_int, x: Complex<c_long>) -> Complex<c_long>;
        safe fn partial_complex_long_long(a: c_int, b: c_int, c: c_int, d: c_int, e: c_int, f: c_int, g: c_int, x: Complex<c_longlong>) -> Complex<c_longlong>;
    }

    let complex = Complex { re: 3.14, im: 6.28 };
    assert_eq!(
        partial_complex_float(
            0x11111111, 0x22222222, 0x33333333, 0x44444444,
            0x55555555, 0x66666666, 0x77777777, complex,
        ),
        complex,
    );

    let complex = Complex { re: 3.14, im: 6.28 };
    assert_eq!(
        partial_complex_double(
            0x11111111, 0x22222222, 0x33333333, 0x44444444,
            0x55555555, 0x66666666, 0x77777777, complex,
        ),
        complex,
    );

    #[cfg(target_arch = "aarch64")]
    {
        let complex = Complex { re: 3.14 as c_longdouble, im: 6.28 as c_longdouble };
        assert_eq!(
            partial_complex_long_double(
                0x11111111, 0x22222222, 0x33333333, 0x44444444,
                0x55555555, 0x66666666, 0x77777777, complex,
            ),
            complex,
        );
    }

    let complex = Complex { re: 3 as c_schar, im: 6 as c_schar };
    assert_eq!(
        partial_complex_char(
            0x11111111, 0x22222222, 0x33333333, 0x44444444,
            0x55555555, 0x66666666, 0x77777777, complex,
        ),
        complex,
    );

    let complex = Complex { re: 3 as c_short, im: 6 as c_short };
    assert_eq!(
        partial_complex_short(
            0x11111111, 0x22222222, 0x33333333, 0x44444444,
            0x55555555, 0x66666666, 0x77777777, complex,
        ),
        complex,
    );

    let complex = Complex { re: 3 as c_int, im: 6 as c_int };
    assert_eq!(
        partial_complex_int(
            0x11111111, 0x22222222, 0x33333333, 0x44444444,
            0x55555555, 0x66666666, 0x77777777, complex,
        ),
        complex,
    );

    let complex = Complex { re: 3 as c_long, im: 6 as c_long };
    assert_eq!(
        partial_complex_long(
            0x11111111, 0x22222222, 0x33333333, 0x44444444,
            0x55555555, 0x66666666, 0x77777777, complex,
        ),
        complex,
    );

    let complex = Complex { re: 3 as c_longlong, im: 6 as c_longlong };
    assert_eq!(
        partial_complex_long_long(
            0x11111111, 0x22222222, 0x33333333, 0x44444444,
            0x55555555, 0x66666666, 0x77777777, complex,
        ),
        complex,
    );
}
