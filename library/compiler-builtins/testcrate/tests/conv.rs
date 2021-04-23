use testcrate::*;

macro_rules! i_to_f {
    ($($from:ty, $into:ty, $fn:ident);*;) => {
        $(
            fuzz(N, |x: $from| {
                let f0 = x as $into;
                let f1: $into = $fn(x);
                // This makes sure that the conversion produced the best rounding possible, and does
                // this independent of `x as $into` rounding correctly.
                // This assumes that float to integer conversion is correct.
                let y_minus_ulp = <$into>::from_bits(f1.to_bits().wrapping_sub(1)) as $from;
                let y = f1 as $from;
                let y_plus_ulp = <$into>::from_bits(f1.to_bits().wrapping_add(1)) as $from;
                let error_minus = <$from as Int>::abs_diff(y_minus_ulp, x);
                let error = <$from as Int>::abs_diff(y, x);
                let error_plus = <$from as Int>::abs_diff(y_plus_ulp, x);
                // The first two conditions check that none of the two closest float values are
                // strictly closer in representation to `x`. The second makes sure that rounding is
                // towards even significand if two float values are equally close to the integer.
                if error_minus < error
                    || error_plus < error
                    || ((error_minus == error || error_plus == error)
                        && ((f0.to_bits() & 1) != 0))
                {
                    panic!(
                        "incorrect rounding by {}({}): {}, ({}, {}, {}), errors ({}, {}, {})",
                        stringify!($fn),
                        x,
                        f1.to_bits(),
                        y_minus_ulp,
                        y,
                        y_plus_ulp,
                        error_minus,
                        error,
                        error_plus,
                    );
                }
                // Test against native conversion. We disable testing on all `x86` because of
                // rounding bugs with `i686`. `powerpc` also has the same rounding bug.
                if f0 != f1 && !cfg!(any(
                    target_arch = "x86",
                    target_arch = "powerpc",
                    target_arch = "powerpc64"
                )) {
                    panic!(
                        "{}({}): std: {}, builtins: {}",
                        stringify!($fn),
                        x,
                        f0,
                        f1,
                    );
                }
            });
        )*
    };
}

#[test]
fn int_to_float() {
    use compiler_builtins::float::conv::{
        __floatdidf, __floatdisf, __floatsidf, __floatsisf, __floattidf, __floattisf,
        __floatundidf, __floatundisf, __floatunsidf, __floatunsisf, __floatuntidf, __floatuntisf,
    };
    use compiler_builtins::int::Int;

    i_to_f!(
        u32, f32, __floatunsisf;
        u32, f64, __floatunsidf;
        i32, f32, __floatsisf;
        i32, f64, __floatsidf;
        u64, f32, __floatundisf;
        u64, f64, __floatundidf;
        i64, f32, __floatdisf;
        i64, f64, __floatdidf;
        u128, f32, __floatuntisf;
        u128, f64, __floatuntidf;
        i128, f32, __floattisf;
        i128, f64, __floattidf;
    );
}

macro_rules! f_to_i {
    ($x:ident, $($f:ty, $fn:ident);*;) => {
        $(
            // it is undefined behavior in the first place to do conversions with NaNs
            if !$x.is_nan() {
                let conv0 = $x as $f;
                let conv1: $f = $fn($x);
                if conv0 != conv1 {
                    panic!("{}({}): std: {}, builtins: {}", stringify!($fn), $x, conv0, conv1);
                }
            }
        )*
    };
}

#[test]
fn float_to_int() {
    use compiler_builtins::float::conv::{
        __fixdfdi, __fixdfsi, __fixdfti, __fixsfdi, __fixsfsi, __fixsfti, __fixunsdfdi,
        __fixunsdfsi, __fixunsdfti, __fixunssfdi, __fixunssfsi, __fixunssfti,
    };

    fuzz_float(N, |x: f32| {
        f_to_i!(x,
            u32, __fixunssfsi;
            u64, __fixunssfdi;
            u128, __fixunssfti;
            i32, __fixsfsi;
            i64, __fixsfdi;
            i128, __fixsfti;
        );
    });
    fuzz_float(N, |x: f64| {
        f_to_i!(x,
            u32, __fixunsdfsi;
            u64, __fixunsdfdi;
            u128, __fixunsdfti;
            i32, __fixdfsi;
            i64, __fixdfdi;
            i128, __fixdfti;
        );
    });
}
