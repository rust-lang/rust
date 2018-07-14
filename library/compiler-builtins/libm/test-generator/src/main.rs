// NOTE we intentionally avoid using the `quote` crate here because it doesn't work with the
// `x86_64-unknown-linux-musl` target.

// NOTE usually the only thing you need to do to test a new math function is to add it to one of the
// macro invocations found in the bottom of this file.

extern crate rand;

use std::error::Error;
use std::fmt::Write as _0;
use std::fs::{self, File};
use std::io::Write as _1;
use std::{i16, u16, u32, u64, u8};

use rand::{Rng, SeedableRng, XorShiftRng};

// Number of test cases to generate
const NTESTS: usize = 10_000;

// TODO tweak these functions to generate edge cases (zero, infinity, NaN) more often
fn f32(rng: &mut XorShiftRng) -> f32 {
    let sign = if rng.gen_bool(0.5) { 1 << 31 } else { 0 };
    let exponent = (rng.gen_range(0, u8::MAX) as u32) << 23;
    let mantissa = rng.gen_range(0, u32::MAX) & ((1 << 23) - 1);

    f32::from_bits(sign + exponent + mantissa)
}

fn f64(rng: &mut XorShiftRng) -> f64 {
    let sign = if rng.gen_bool(0.5) { 1 << 63 } else { 0 };
    let exponent = (rng.gen_range(0, u16::MAX) as u64 & ((1 << 11) - 1)) << 52;
    let mantissa = rng.gen_range(0, u64::MAX) & ((1 << 52) - 1);

    f64::from_bits(sign + exponent + mantissa)
}

// fn(f32) -> f32
macro_rules! f32_f32 {
    ($($intr:ident,)*) => {
        fn f32_f32(rng: &mut XorShiftRng) -> Result<(), Box<Error>> {
            // MUSL C implementation of the function to test
            extern "C" {
                $(fn $intr(_: f32) -> f32;)*
            }

            $(
                let mut cases = String::new();
                for _ in 0..NTESTS {
                    let inp = f32(rng);
                    let out = unsafe { $intr(inp) };

                    let inp = inp.to_bits();
                    let out = out.to_bits();

                    write!(cases, "({}, {})", inp, out).unwrap();
                    cases.push(',');
                }

                let mut f = File::create(concat!("tests/", stringify!($intr), ".rs"))?;
                write!(f, "
                    #![deny(warnings)]

                    extern crate libm;

                    use std::panic;

                    #[test]
                    fn {0}() {{
                        const CASES: &[(u32, u32)] = &[
                            {1}
                        ];

                        for case in CASES {{
                            let (inp, expected) = *case;

                            if let Ok(outf) =
                                panic::catch_unwind(|| libm::{0}(f32::from_bits(inp)))
                            {{
                                let outi = outf.to_bits();

                                if !((outf.is_nan() && f32::from_bits(expected).is_nan())
                                    || libm::_eqf(outi, expected))
                                {{
                                    panic!(
                                        \"input: {{}}, output: {{}}, expected: {{}}\",
                                        inp, outi, expected,
                                    );
                                }}
                            }} else {{
                                panic!(
                                    \"input: {{}}, output: PANIC, expected: {{}}\",
                                    inp, expected,
                                );
                            }}
                        }}
                    }}
",
                       stringify!($intr),
                       cases)?;
            )*

            Ok(())
        }
    }
}

// fn(f32, f32) -> f32
macro_rules! f32f32_f32 {
    ($($intr:ident,)*) => {
        fn f32f32_f32(rng: &mut XorShiftRng) -> Result<(), Box<Error>> {
            extern "C" {
                $(fn $intr(_: f32, _: f32) -> f32;)*
            }

            $(
                let mut cases = String::new();
                for _ in 0..NTESTS {
                    let i1 = f32(rng);
                    let i2 = f32(rng);
                    let out = unsafe { $intr(i1, i2) };

                    let i1 = i1.to_bits();
                    let i2 = i2.to_bits();
                    let out = out.to_bits();

                    write!(cases, "(({}, {}), {})", i1, i2, out).unwrap();
                    cases.push(',');
                }

                let mut f = File::create(concat!("tests/", stringify!($intr), ".rs"))?;
                write!(f, "
                    #![deny(warnings)]

                    extern crate libm;

                    use std::panic;

                    #[test]
                    fn {0}() {{
                        const CASES: &[((u32, u32), u32)] = &[
                            {1}
                        ];

                        for case in CASES {{
                            let ((i1, i2), expected) = *case;

                            if let Ok(outf) = panic::catch_unwind(|| {{
                                libm::{0}(f32::from_bits(i1), f32::from_bits(i2))
                            }}) {{
                                let outi = outf.to_bits();

                                if !((outf.is_nan() && f32::from_bits(expected).is_nan())
                                    || libm::_eqf(outi, expected))
                                {{
                                    panic!(
                                        \"input: {{:?}}, output: {{}}, expected: {{}}\",
                                        (i1, i2),
                                        outi,
                                        expected,
                                    );
                                }}
                            }} else {{
                                panic!(
                                    \"input: {{:?}}, output: PANIC, expected: {{}}\",
                                    (i1, i2),
                                    expected,
                                );
                            }}
                        }}
                    }}
",
                       stringify!($intr),
                       cases)?;
            )*

            Ok(())
        }
    };
}

// fn(f32, f32, f32) -> f32
macro_rules! f32f32f32_f32 {
    ($($intr:ident,)*) => {
        fn f32f32f32_f32(rng: &mut XorShiftRng) -> Result<(), Box<Error>> {
            extern "C" {
                $(fn $intr(_: f32, _: f32, _: f32) -> f32;)*
            }

            $(
                let mut cases = String::new();
                for _ in 0..NTESTS {
                    let i1 = f32(rng);
                    let i2 = f32(rng);
                    let i3 = f32(rng);
                    let out = unsafe { $intr(i1, i2, i3) };

                    let i1 = i1.to_bits();
                    let i2 = i2.to_bits();
                    let i3 = i3.to_bits();
                    let out = out.to_bits();

                    write!(cases, "(({}, {}, {}), {})", i1, i2, i3, out).unwrap();
                    cases.push(',');
                }

                let mut f = File::create(concat!("tests/", stringify!($intr), ".rs"))?;
                write!(f, "
                    #![deny(warnings)]

                    extern crate libm;

                    use std::panic;

                    #[test]
                    fn {0}() {{
                        const CASES: &[((u32, u32, u32), u32)] = &[
                            {1}
                        ];

                        for case in CASES {{
                            let ((i1, i2, i3), expected) = *case;

                            if let Ok(outf) = panic::catch_unwind(|| {{
                                libm::{0}(
                                    f32::from_bits(i1),
                                    f32::from_bits(i2),
                                    f32::from_bits(i3),
                                )
                            }}) {{
                                let outi = outf.to_bits();

                                if !((outf.is_nan() && f32::from_bits(expected).is_nan())
                                    || libm::_eqf(outi, expected))
                                {{
                                    panic!(
                                        \"input: {{:?}}, output: {{}}, expected: {{}}\",
                                        (i1, i2, i3),
                                        outi,
                                        expected,
                                    );
                                }}
                            }} else {{
                                panic!(
                                    \"input: {{:?}}, output: PANIC, expected: {{}}\",
                                    (i1, i2, i3),
                                    expected,
                                );
                            }}
                        }}
                    }}
",
                       stringify!($intr),
                       cases)?;
            )*

            Ok(())
        }
    };
}

// fn(f32, i32) -> f32
macro_rules! f32i32_f32 {
    ($($intr:ident,)*) => {
        fn f32i32_f32(rng: &mut XorShiftRng) -> Result<(), Box<Error>> {
            extern "C" {
                $(fn $intr(_: f32, _: i32) -> f32;)*
            }

            $(
                let mut cases = String::new();
                for _ in 0..NTESTS {
                    let i1 = f32(rng);
                    let i2 = rng.gen_range(i16::MIN, i16::MAX);
                    let out = unsafe { $intr(i1, i2 as i32) };

                    let i1 = i1.to_bits();
                    let out = out.to_bits();

                    write!(cases, "(({}, {}), {})", i1, i2, out).unwrap();
                    cases.push(',');
                }

                let mut f = File::create(concat!("tests/", stringify!($intr), ".rs"))?;
                write!(f, "
                    #![deny(warnings)]

                    extern crate libm;

                    use std::panic;

                    #[test]
                    fn {0}() {{
                        const CASES: &[((u32, i16), u32)] = &[
                            {1}
                        ];

                        for case in CASES {{
                            let ((i1, i2), expected) = *case;

                            if let Ok(outf) = panic::catch_unwind(|| {{
                                libm::{0}(f32::from_bits(i1), i2 as i32)
                            }}) {{
                                let outi = outf.to_bits();

                                if !((outf.is_nan() && f32::from_bits(expected).is_nan())
                                    || libm::_eqf(outi, expected))
                                {{
                                    panic!(
                                        \"input: {{:?}}, output: {{}}, expected: {{}}\",
                                        (i1, i2),
                                        outi,
                                        expected,
                                    );
                                }}
                            }} else {{
                                panic!(
                                    \"input: {{:?}}, output: PANIC, expected: {{}}\",
                                    (i1, i2),
                                    expected,
                                );
                            }}
                        }}
                    }}
",
                       stringify!($intr),
                       cases)?;
            )*

            Ok(())
        }
    };
}

// fn(f64) -> f64
macro_rules! f64_f64 {
    ($($intr:ident,)*) => {
        fn f64_f64(rng: &mut XorShiftRng) -> Result<(), Box<Error>> {
            // MUSL C implementation of the function to test
            extern "C" {
                $(fn $intr(_: f64) -> f64;)*
            }

            $(
                let mut cases = String::new();
                for _ in 0..NTESTS {
                    let inp = f64(rng);
                    let out = unsafe { $intr(inp) };

                    let inp = inp.to_bits();
                    let out = out.to_bits();

                    write!(cases, "({}, {})", inp, out).unwrap();
                    cases.push(',');
                }

                let mut f = File::create(concat!("tests/", stringify!($intr), ".rs"))?;
                write!(f, "
                    #![deny(warnings)]

                    extern crate libm;

                    use std::panic;

                    #[test]
                    fn {0}() {{
                        const CASES: &[(u64, u64)] = &[
                            {1}
                        ];

                        for case in CASES {{
                            let (inp, expected) = *case;

                            if let Ok(outf) = panic::catch_unwind(|| {{
                                libm::{0}(f64::from_bits(inp))
                            }}) {{
                                let outi = outf.to_bits();

                                if !((outf.is_nan() && f64::from_bits(expected).is_nan())
                                    || libm::_eq(outi, expected))
                                {{
                                    panic!(
                                        \"input: {{}}, output: {{}}, expected: {{}}\",
                                        inp,
                                        outi,
                                        expected,
                                    );
                                }}
                            }} else {{
                                panic!(
                                    \"input: {{}}, output: PANIC, expected: {{}}\",
                                    inp,
                                    expected,
                                );
                            }}
                        }}
                    }}
",
                       stringify!($intr),
                       cases)?;
            )*

            Ok(())
        }
    }
}

// fn(f64, f64) -> f64
macro_rules! f64f64_f64 {
    ($($intr:ident,)*) => {
        fn f64f64_f64(rng: &mut XorShiftRng) -> Result<(), Box<Error>> {
            extern "C" {
                $(fn $intr(_: f64, _: f64) -> f64;)*
            }

            $(
                let mut cases = String::new();
                for _ in 0..NTESTS {
                    let i1 = f64(rng);
                    let i2 = f64(rng);
                    let out = unsafe { $intr(i1, i2) };

                    let i1 = i1.to_bits();
                    let i2 = i2.to_bits();
                    let out = out.to_bits();

                    write!(cases, "(({}, {}), {})", i1, i2, out).unwrap();
                    cases.push(',');
                }

                let mut f = File::create(concat!("tests/", stringify!($intr), ".rs"))?;
                write!(f, "
                    #![deny(warnings)]

                    extern crate libm;

                    use std::panic;

                    #[test]
                    fn {0}() {{
                        const CASES: &[((u64, u64), u64)] = &[
                            {1}
                        ];

                        for case in CASES {{
                            let ((i1, i2), expected) = *case;

                            if let Ok(outf) = panic::catch_unwind(|| {{
                                libm::{0}(f64::from_bits(i1), f64::from_bits(i2))
                            }}) {{
                                let outi = outf.to_bits();

                                if !((outf.is_nan() && f64::from_bits(expected).is_nan()) ||
                                    libm::_eq(outi, expected)) {{
                                    panic!(
                                        \"input: {{:?}}, output: {{}}, expected: {{}}\",
                                        (i1, i2),
                                        outi,
                                        expected,
                                    );
                                }}
                            }} else {{
                                panic!(
                                    \"input: {{:?}}, output: PANIC, expected: {{}}\",
                                    (i1, i2),
                                    expected,
                                );
                            }}
                        }}
                    }}
",
                       stringify!($intr),
                       cases)?;
            )*

            Ok(())
        }
    };
}

// fn(f64, f64, f64) -> f64
macro_rules! f64f64f64_f64 {
    ($($intr:ident,)*) => {
        fn f64f64f64_f64(rng: &mut XorShiftRng) -> Result<(), Box<Error>> {
            extern "C" {
                $(fn $intr(_: f64, _: f64, _: f64) -> f64;)*
            }

            $(
                let mut cases = String::new();
                for _ in 0..NTESTS {
                    let i1 = f64(rng);
                    let i2 = f64(rng);
                    let i3 = f64(rng);
                    let out = unsafe { $intr(i1, i2, i3) };

                    let i1 = i1.to_bits();
                    let i2 = i2.to_bits();
                    let i3 = i3.to_bits();
                    let out = out.to_bits();

                    write!(cases, "(({}, {}, {}), {})", i1, i2, i3, out).unwrap();
                    cases.push(',');
                }

                let mut f = File::create(concat!("tests/", stringify!($intr), ".rs"))?;
                write!(f, "
                    #![deny(warnings)]

                    extern crate libm;

                    use std::panic;

                    #[test]
                    fn {0}() {{
                        const CASES: &[((u64, u64, u64), u64)] = &[
                            {1}
                        ];

                        for case in CASES {{
                            let ((i1, i2, i3), expected) = *case;

                            if let Ok(outf) = panic::catch_unwind(|| {{
                                libm::{0}(
                                    f64::from_bits(i1),
                                    f64::from_bits(i2),
                                    f64::from_bits(i3),
                                )
                            }}) {{
                                let outi = outf.to_bits();

                                if !((outf.is_nan() && f64::from_bits(expected).is_nan())
                                    || libm::_eq(outi, expected))
                                {{
                                    panic!(
                                        \"input: {{:?}}, output: {{}}, expected: {{}}\",
                                        (i1, i2, i3),
                                        outi,
                                        expected,
                                    );
                                }}
                            }} else {{
                                panic!(
                                    \"input: {{:?}}, output: PANIC, expected: {{}}\",
                                    (i1, i2, i3),
                                    expected,
                                );
                            }}
                        }}
                    }}
",
                       stringify!($intr),
                       cases)?;
            )*

            Ok(())
        }
    };
}

// fn(f64, i32) -> f64
macro_rules! f64i32_f64 {
    ($($intr:ident,)*) => {
        fn f64i32_f64(rng: &mut XorShiftRng) -> Result<(), Box<Error>> {
            extern "C" {
                $(fn $intr(_: f64, _: i32) -> f64;)*
            }

            $(
                let mut cases = String::new();
                for _ in 0..NTESTS {
                    let i1 = f64(rng);
                    let i2 = rng.gen_range(i16::MIN, i16::MAX);
                    let out = unsafe { $intr(i1, i2 as i32) };

                    let i1 = i1.to_bits();
                    let out = out.to_bits();

                    write!(cases, "(({}, {}), {})", i1, i2, out).unwrap();
                    cases.push(',');
                }

                let mut f = File::create(concat!("tests/", stringify!($intr), ".rs"))?;
                write!(f, "
                    #![deny(warnings)]

                    extern crate libm;

                    use std::panic;

                    #[test]
                    fn {0}() {{
                        const CASES: &[((u64, i16), u64)] = &[
                            {1}
                        ];

                        for case in CASES {{
                            let ((i1, i2), expected) = *case;

                            if let Ok(outf) = panic::catch_unwind(|| {{
                                libm::{0}(f64::from_bits(i1), i2 as i32)
                            }}) {{
                                let outi = outf.to_bits();

                                if !((outf.is_nan() && f64::from_bits(expected).is_nan()) ||
                                    libm::_eq(outi, expected)) {{
                                    panic!(
                                        \"input: {{:?}}, output: {{}}, expected: {{}}\",
                                        (i1, i2),
                                        outi,
                                        expected,
                                    );
                                }}
                            }} else {{
                                panic!(
                                    \"input: {{:?}}, output: PANIC, expected: {{}}\",
                                    (i1, i2),
                                    expected,
                                );
                            }}
                        }}
                    }}
",
                       stringify!($intr),
                       cases)?;
            )*

            Ok(())
        }
    };
}

fn main() -> Result<(), Box<Error>> {
    fs::remove_dir_all("tests").ok();
    fs::create_dir("tests")?;

    let mut rng = XorShiftRng::from_rng(&mut rand::thread_rng())?;

    f32_f32(&mut rng)?;
    f32f32_f32(&mut rng)?;
    f32f32f32_f32(&mut rng)?;
    f32i32_f32(&mut rng)?;
    f64_f64(&mut rng)?;
    f64f64_f64(&mut rng)?;
    f64f64f64_f64(&mut rng)?;
    f64i32_f64(&mut rng)?;

    Ok(())
}

/* Functions to test */

// With signature `fn(f32) -> f32`
f32_f32! {
    // acosf,
    floorf,
    truncf,
    // asinf,
    // atanf,
    // cbrtf,
    ceilf,
    // cosf,
    // coshf,
    // exp2f,
    expf,
    // fdimf,
    // log10f,
    // log2f,
    logf,
    roundf,
    // sinf,
    // sinhf,
    // tanf,
    // tanhf,
    fabsf,
    sqrtf,
}

// With signature `fn(f32, f32) -> f32`
f32f32_f32! {
    // atan2f,
    hypotf,
    fmodf,
    powf,
}

// With signature `fn(f32, f32, f32) -> f32`
f32f32f32_f32! {
    // fmaf,
}

// With signature `fn(f32, i32) -> f32`
f32i32_f32! {
    scalbnf,
}

// With signature `fn(f64) -> f64`
f64_f64! {
    // acos,
    // asin,
    // atan,
    // cbrt,
    // ceil,
    // cos,
    // cosh,
    // exp,
    // exp2,
    // expm1,
    floor,
    // log,
    // log10,
    // log1p,
    // log2,
    round,
    // sin,
    // sinh,
    sqrt,
    // tan,
    // tanh,
    trunc,
    fabs,
}

// With signature `fn(f64, f64) -> f64`
f64f64_f64! {
    // atan2,
    // fdim,
    fmod,
    hypot,
    // pow,
}

// With signature `fn(f64, f64, f64) -> f64`
f64f64f64_f64! {
    // fma,
}

// With signature `fn(f64, i32) -> f64`
f64i32_f64! {
    scalbn,
}
