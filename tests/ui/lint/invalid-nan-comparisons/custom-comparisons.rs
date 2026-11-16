//@ check-pass

#![deny(invalid_nan_comparisons)]

use std::cmp::Ordering;

struct NotFloat;

impl PartialEq<f64> for NotFloat {
    fn eq(&self, other: &f64) -> bool {
        other.is_nan()
    }
}

impl PartialEq<NotFloat> for f64 {
    fn eq(&self, other: &NotFloat) -> bool {
        other == self
    }
}

impl PartialOrd<f64> for NotFloat {
    fn partial_cmp(&self, other: &f64) -> Option<Ordering> {
        other.is_nan().then_some(Ordering::Equal)
    }
}

impl PartialOrd<NotFloat> for f64 {
    fn partial_cmp(&self, other: &NotFloat) -> Option<Ordering> {
        other.partial_cmp(self).map(Ordering::reverse)
    }
}

fn main() {
    let value = NotFloat;
    let _ = value == f64::NAN;
    let _ = value != f64::NAN;
    let _ = value < f64::NAN;
    let _ = value <= f64::NAN;
    let _ = value > f64::NAN;
    let _ = value >= f64::NAN;

    let _ = f64::NAN == value;
    let _ = f64::NAN != value;
    let _ = f64::NAN < value;
    let _ = f64::NAN <= value;
    let _ = f64::NAN > value;
    let _ = f64::NAN >= value;

    let _ = &value == &f64::NAN;
    let _ = &value != &f64::NAN;
    let _ = &value < &f64::NAN;
    let _ = &value <= &f64::NAN;
    let _ = &value > &f64::NAN;
    let _ = &value >= &f64::NAN;

    let _ = &f64::NAN == &value;
    let _ = &f64::NAN != &value;
    let _ = &f64::NAN < &value;
    let _ = &f64::NAN <= &value;
    let _ = &f64::NAN > &value;
    let _ = &f64::NAN >= &value;

    generic(value);
}

fn generic<T: PartialOrd<f64>>(value: T) {
    let _ = value == f64::NAN;
    let _ = value != f64::NAN;
    let _ = value < f64::NAN;
    let _ = value <= f64::NAN;
    let _ = value > f64::NAN;
    let _ = value >= f64::NAN;
}
