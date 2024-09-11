//@ compile-flags: -Zmir-opt-level=0 -Znext-solver
//@ run-pass
// ignore-tidy-linelength

// This tests the float classification functions, for regular runtime code and for const evaluation.

#![feature(f16_const)]
#![feature(f128_const)]
#![feature(const_float_classify)]

use std::hint::black_box;
use std::num::FpCategory::*;

macro_rules! both_assert {
    ($a:expr, NonDet) => {
        {
            // Compute `a`, but do not compare with anything as the result is non-deterministic.
            const _: () = { let _val = $a; };
            // `black_box` prevents promotion, and MIR opts are disabled above, so this is truly
            // going through LLVM.
            let _val = black_box($a);
        }
    };
    ($a:expr, $b:ident) => {
        {
            const _: () = assert!(matches!($a, $b));
            assert!(black_box($a) == black_box($b));
        }
    };
}

macro_rules! suite {
    ( $tyname:ident: $( $tt:tt )* ) => {
        fn f32() {
            type $tyname = f32;
            suite_inner!(f32 $($tt)*);
        }

        fn f64() {
            type $tyname = f64;
            suite_inner!(f64 $($tt)*);
        }
    }
}

macro_rules! suite_inner {
    (
        $ty:ident [$( $fn:ident ),*]
        $val:expr => [$($out:ident),*]

        $( $tail:tt )*
    ) => {
        $( both_assert!($ty::$fn($val), $out); )*
        suite_inner!($ty [$($fn),*] $($tail)*)
    };

    ( $ty:ident [$( $fn:ident ),*]) => {};
}

// The result of the `is_sign` methods are not checked for correctness, since we do not
// guarantee anything about the signedness of NaNs. See
// https://rust-lang.github.io/rfcs/3514-float-semantics.html.

suite! { T: // type alias for the type we are testing
                   [ classify, is_nan, is_infinite, is_finite, is_normal, is_sign_positive, is_sign_negative]
     -0.0 / 0.0 => [      Nan,   true,       false,     false,     false,           NonDet,           NonDet]
      0.0 / 0.0 => [      Nan,   true,       false,     false,     false,           NonDet,           NonDet]
            1.0 => [   Normal,  false,       false,      true,      true,             true,            false]
           -1.0 => [   Normal,  false,       false,      true,      true,            false,             true]
            0.0 => [     Zero,  false,       false,      true,     false,             true,            false]
           -0.0 => [     Zero,  false,       false,      true,     false,            false,             true]
      1.0 / 0.0 => [ Infinite,  false,        true,     false,     false,             true,            false]
     -1.0 / 0.0 => [ Infinite,  false,        true,     false,     false,            false,             true]
   1.0 / T::MAX => [Subnormal,  false,       false,      true,     false,             true,            false]
  -1.0 / T::MAX => [Subnormal,  false,       false,      true,     false,            false,             true]
}

fn main() {
    f32();
    f64();
    // FIXME(f16_f128): also test f16 and f128
}
