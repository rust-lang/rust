// Test precise capture of a multi-variant enum (when remaining variants are
// visibly uninhabited).
// revisions: min_exhaustive_patterns exhaustive_patterns
// edition:2021
// run-pass
#![cfg_attr(exhaustive_patterns, feature(exhaustive_patterns))]
#![cfg_attr(min_exhaustive_patterns, feature(min_exhaustive_patterns))]
//[min_exhaustive_patterns]~^ WARN the feature `min_exhaustive_patterns` is incomplete
#![feature(never_type)]

pub fn main() {
    let mut r = Result::<!, (u32, u32)>::Err((0, 0));
    let mut f = || {
        let Err((ref mut a, _)) = r;
        *a = 1;
    };
    let mut g = || {
        let Err((_, ref mut b)) = r;
        *b = 2;
    };
    f();
    g();
    assert_eq!(r, Err((1, 2)));
}
