//@ run-pass
//@ revisions: opt noopt ctfe
//@[opt] compile-flags: -O
//@[noopt] compile-flags: -Zmir-opt-level=0
// ignore-tidy-linelength

// This tests the float classification functions, for regular runtime code and for const evaluation.

#![feature(f16)]
#![feature(f128)]

use std::num::FpCategory::*;

#[cfg(not(ctfe))]
use std::hint::black_box;
#[cfg(ctfe)]
#[allow(unused)]
const fn black_box<T>(x: T) -> T { x }

#[cfg(not(ctfe))]
macro_rules! assert_test {
    ($a:expr, NonDet) => {
        {
            // Compute `a`, but do not compare with anything as the result is non-deterministic.
            let _val = $a;
        }
    };
    ($a:expr, $b:ident) => {
        {
            // Let-bind to avoid promotion.
            // No black_box here! That can mask x87 failures.
            let a = $a;
            let b = $b;
            assert_eq!(a, b, "{} produces wrong result", stringify!($a));
        }
    };
}
#[cfg(ctfe)]
macro_rules! assert_test {
    ($a:expr, NonDet) => {
        {
            // Compute `a`, but do not compare with anything as the result is non-deterministic.
            const _: () = { let _val = $a; };
        }
    };
    ($a:expr, $b:ident) => {
        {
            const _: () = assert!(matches!($a, $b));
        }
    };
}

macro_rules! suite {
    ( $tyname:ident => $( $tt:tt )* ) => {
        fn f32() {
            #[allow(unused)]
            type $tyname = f32;
            suite_inner!(f32 => $($tt)*);
        }

        fn f64() {
            #[allow(unused)]
            type $tyname = f64;
            suite_inner!(f64 => $($tt)*);
        }
    }
}

macro_rules! suite_inner {
    (
        $ty:ident => [$( $fn:ident ),*]:
        $(@cfg: $attr:meta)?
        $val:expr => [$($out:ident),*],

        $( $tail:tt )*
    ) => {
        $(#[cfg($attr)])?
        {
            // No black_box here! That can mask x87 failures.
            $( assert_test!($ty::$fn($val), $out); )*
        }
        suite_inner!($ty => [$($fn),*]: $($tail)*)
    };

    ( $ty:ident => [$( $fn:ident ),*]:) => {};
}

// The result of the `is_sign` methods are not checked for correctness, since we do not
// guarantee anything about the signedness of NaNs. See
// https://rust-lang.github.io/rfcs/3514-float-semantics.html.

suite! { T => // type alias for the type we are testing
                    [ classify, is_nan, is_infinite, is_finite, is_normal, is_sign_positive, is_sign_negative]:
    black_box(0.0) / black_box(0.0) =>
                    [      Nan,   true,       false,     false,     false,           NonDet,           NonDet],
    black_box(0.0) / black_box(-0.0) =>
                    [      Nan,   true,       false,     false,     false,           NonDet,           NonDet],
    black_box(0.0) * black_box(T::INFINITY) =>
                    [      Nan,   true,       false,     false,     false,           NonDet,           NonDet],
    black_box(0.0) * black_box(T::NEG_INFINITY) =>
                    [      Nan,   true,       false,     false,     false,           NonDet,           NonDet],
             1.0 => [   Normal,  false,       false,      true,      true,             true,            false],
            -1.0 => [   Normal,  false,       false,      true,      true,            false,             true],
             0.0 => [     Zero,  false,       false,      true,     false,             true,            false],
            -0.0 => [     Zero,  false,       false,      true,     false,            false,             true],
    1.0 / black_box(0.0) =>
                    [ Infinite,  false,        true,     false,     false,             true,            false],
    -1.0 / black_box(0.0) =>
                    [ Infinite,  false,        true,     false,     false,            false,             true],
    2.0 * black_box(T::MAX) =>
                    [ Infinite,  false,        true,     false,     false,             true,            false],
    -2.0 * black_box(T::MAX) =>
                    [ Infinite,  false,        true,     false,     false,            false,             true],
    1.0 / black_box(T::MAX) =>
                    [Subnormal,  false,       false,      true,     false,             true,            false],
   -1.0 / black_box(T::MAX) =>
                    [Subnormal,  false,       false,      true,     false,            false,             true],
    // This specific expression causes trouble on x87 due to
    // <https://github.com/rust-lang/rust/issues/114479>.
    @cfg: not(all(target_arch = "x86", not(target_feature = "sse2")))
    { let x = black_box(T::MAX); x * x } =>
                    [ Infinite,  false,        true,     false,     false,             true,            false],
}

fn main() {
    f32();
    f64();
    // FIXME(f16_f128): also test f16 and f128
}
