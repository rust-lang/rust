// run-pass
#![feature(test)]
#![allow(invalid_nan_comparisons)]

extern crate test;

use std::mem::size_of;
use test::black_box;

// Ensure the const-eval result and runtime result of float comparison are equivalent.

macro_rules! compare {
    ($op:tt) => {
        compare!(
            [f64::NEG_INFINITY, -f64::MAX, -1.0, -0.0, 0.0, 1.0, f64::MAX, f64::INFINITY, f64::NAN],
            $op
        );
    };
    ([$($lhs:expr),+], $op:tt) => {
        $(compare!(
            $lhs,
            $op,
            [f64::NEG_INFINITY, -f64::MAX, -1.0, -0.0, 0.0, 1.0, f64::MAX, f64::INFINITY, f64::NAN]
        );)+
    };
    ($lhs:expr, $op:tt, [$($rhs:expr),+]) => {
        $({
            // Wrap the check in its own function to reduce time needed to borrowck.
            fn check() {
                static CONST_EVAL: bool = $lhs $op $rhs;
                let runtime_eval = black_box($lhs) $op black_box($rhs);
                assert_eq!(CONST_EVAL, runtime_eval, stringify!($lhs $op $rhs));
                assert_eq!(
                    size_of::<[u8; ($lhs $op $rhs) as usize]>(),
                    runtime_eval as usize,
                    stringify!($lhs $op $rhs (forced const eval))
                );
            }
            check();
        })+
    };
}

fn main() {
    assert_eq!(0.0/0.0 < 0.0/0.0, false);
    assert_eq!(0.0/0.0 > 0.0/0.0, false);
    assert_eq!(f64::NAN < f64::NAN, false);
    assert_eq!(f64::NAN > f64::NAN, false);

    compare!(==);
    compare!(!=);
    compare!(<);
    compare!(<=);
    compare!(>);
    compare!(>=);
}
