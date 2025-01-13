//! Helper CLI utility for common tasks.

#![cfg_attr(f16_enabled, feature(f16))]
#![cfg_attr(f128_enabled, feature(f128))]

use std::any::type_name;
use std::env;
use std::str::FromStr;

#[cfg(feature = "build-mpfr")]
use az::Az;
#[cfg(feature = "build-mpfr")]
use libm_test::mpfloat::MpOp;
use libm_test::{MathOp, TupleCall};

const USAGE: &str = "\
usage:

cargo run -p util -- <SUBCOMMAND>

SUBCOMMAND:
    eval <BASIS> <OP> inputs...
        Evaulate the expression with a given basis. This can be useful for
        running routines with a debugger, or quickly checking input. Examples:
        * eval musl sinf 1.234 # print the results of musl sinf(1.234f32)
        * eval mpfr pow 1.234 2.432 # print the results of mpfr pow(1.234, 2.432)
";

fn main() {
    let args = env::args().collect::<Vec<_>>();
    let str_args = args.iter().map(|s| s.as_str()).collect::<Vec<_>>();

    match &str_args.as_slice()[1..] {
        ["eval", basis, op, inputs @ ..] => do_eval(basis, op, inputs),
        _ => {
            println!("{USAGE}\nunrecognized input `{str_args:?}`");
            std::process::exit(1);
        }
    }
}

macro_rules! handle_call {
    (
        fn_name: $fn_name:ident,
        CFn: $CFn:ty,
        RustFn: $RustFn:ty,
        RustArgs: $RustArgs:ty,
        attrs: [$($attr:meta),*],
        extra: ($basis:ident, $op:ident, $inputs:ident),
        fn_extra: $musl_fn:expr,
    ) => {
        $(#[$attr])*
        if $op == stringify!($fn_name) {
            type Op = libm_test::op::$fn_name::Routine;

            let input = <$RustArgs>::parse($inputs);
            let libm_fn: <Op as MathOp>::RustFn = libm::$fn_name;

            let output = match $basis {
                "libm" => input.call(libm_fn),
                #[cfg(feature = "build-musl")]
                "musl" => {
                    let musl_fn: <Op as MathOp>::CFn =
                        $musl_fn.unwrap_or_else(|| panic!("no musl function for {}", $op));
                    input.call(musl_fn)
                }
                #[cfg(feature = "build-mpfr")]
                "mpfr" => {
                    let mut mp = <Op as MpOp>::new_mp();
                    Op::run(&mut mp, input)
                }
                _ => panic!("unrecognized or disabled basis '{}'", $basis),
            };
            println!("{output:?}");
            return;
        }
    };
}

/// Evaluate the specified operation with a given basis.
fn do_eval(basis: &str, op: &str, inputs: &[&str]) {
    libm_macros::for_each_function! {
        callback: handle_call,
        emit_types: [CFn, RustFn, RustArgs],
        extra: (basis, op, inputs),
        fn_extra: match MACRO_FN_NAME {
            copysignf16 | copysignf128 | fabsf16 | fabsf128 | truncf16 | truncf128  => None,
            _ => Some(musl_math_sys::MACRO_FN_NAME)
        }
    }

    panic!("no operation matching {op}");
}

/// Parse a tuple from a space-delimited string.
trait ParseTuple {
    fn parse(input: &[&str]) -> Self;
}

macro_rules! impl_parse_tuple {
    ($ty:ty) => {
        impl ParseTuple for ($ty,) {
            fn parse(input: &[&str]) -> Self {
                assert_eq!(input.len(), 1, "expected a single argument, got {input:?}");
                (parse(input, 0),)
            }
        }

        impl ParseTuple for ($ty, $ty) {
            fn parse(input: &[&str]) -> Self {
                assert_eq!(input.len(), 2, "expected two arguments, got {input:?}");
                (parse(input, 0), parse(input, 1))
            }
        }

        impl ParseTuple for ($ty, i32) {
            fn parse(input: &[&str]) -> Self {
                assert_eq!(input.len(), 2, "expected two arguments, got {input:?}");
                (parse(input, 0), parse(input, 1))
            }
        }

        impl ParseTuple for (i32, $ty) {
            fn parse(input: &[&str]) -> Self {
                assert_eq!(input.len(), 2, "expected two arguments, got {input:?}");
                (parse(input, 0), parse(input, 1))
            }
        }

        impl ParseTuple for ($ty, $ty, $ty) {
            fn parse(input: &[&str]) -> Self {
                assert_eq!(input.len(), 2, "expected three arguments, got {input:?}");
                (parse(input, 0), parse(input, 1), parse(input, 3))
            }
        }
    };
}

#[allow(unused_macros)]
#[cfg(feature = "build-mpfr")]
macro_rules! impl_parse_tuple_via_rug {
    ($ty:ty) => {
        impl ParseTuple for ($ty,) {
            fn parse(input: &[&str]) -> Self {
                assert_eq!(input.len(), 1, "expected a single argument, got {input:?}");
                (parse_rug(input, 0),)
            }
        }

        impl ParseTuple for ($ty, $ty) {
            fn parse(input: &[&str]) -> Self {
                assert_eq!(input.len(), 2, "expected two arguments, got {input:?}");
                (parse_rug(input, 0), parse_rug(input, 1))
            }
        }

        impl ParseTuple for ($ty, i32) {
            fn parse(input: &[&str]) -> Self {
                assert_eq!(input.len(), 2, "expected two arguments, got {input:?}");
                (parse_rug(input, 0), parse(input, 1))
            }
        }

        impl ParseTuple for (i32, $ty) {
            fn parse(input: &[&str]) -> Self {
                assert_eq!(input.len(), 2, "expected two arguments, got {input:?}");
                (parse(input, 0), parse_rug(input, 1))
            }
        }

        impl ParseTuple for ($ty, $ty, $ty) {
            fn parse(input: &[&str]) -> Self {
                assert_eq!(input.len(), 2, "expected three arguments, got {input:?}");
                (parse_rug(input, 0), parse_rug(input, 1), parse_rug(input, 3))
            }
        }
    };
}

// Fallback for when Rug is not built.
#[allow(unused_macros)]
#[cfg(not(feature = "build-mpfr"))]
macro_rules! impl_parse_tuple_via_rug {
    ($ty:ty) => {
        impl ParseTuple for ($ty,) {
            fn parse(_input: &[&str]) -> Self {
                panic!("parsing this type requires the `build-mpfr` feature")
            }
        }

        impl ParseTuple for ($ty, $ty) {
            fn parse(_input: &[&str]) -> Self {
                panic!("parsing this type requires the `build-mpfr` feature")
            }
        }

        impl ParseTuple for ($ty, i32) {
            fn parse(_input: &[&str]) -> Self {
                panic!("parsing this type requires the `build-mpfr` feature")
            }
        }

        impl ParseTuple for (i32, $ty) {
            fn parse(_input: &[&str]) -> Self {
                panic!("parsing this type requires the `build-mpfr` feature")
            }
        }

        impl ParseTuple for ($ty, $ty, $ty) {
            fn parse(_input: &[&str]) -> Self {
                panic!("parsing this type requires the `build-mpfr` feature")
            }
        }
    };
}

impl_parse_tuple!(f32);
impl_parse_tuple!(f64);

#[cfg(f16_enabled)]
impl_parse_tuple_via_rug!(f16);
#[cfg(f128_enabled)]
impl_parse_tuple_via_rug!(f128);

/// Try to parse the number, printing a nice message on failure.
fn parse<F: FromStr>(input: &[&str], idx: usize) -> F {
    let s = input[idx];
    s.parse().unwrap_or_else(|_| panic!("invalid {} input '{s}'", type_name::<F>()))
}

/// Try to parse the float type going via `rug`, for `f16` and `f128` which don't yet implement
/// `FromStr`.
#[cfg(feature = "build-mpfr")]
fn parse_rug<F: libm_test::Float>(input: &[&str], idx: usize) -> F
where
    rug::Float: az::Cast<F>,
{
    let s = input[idx];
    let x =
        rug::Float::parse(s).unwrap_or_else(|_| panic!("invalid {} input '{s}'", type_name::<F>()));
    let x = rug::Float::with_val(F::BITS, x);
    x.az()
}
