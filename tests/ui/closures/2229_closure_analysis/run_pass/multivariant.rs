// Test precise capture of a multi-variant enum (when remaining variants are
// visibly uninhabited).
// edition:2021
// run-pass
#![feature(exhaustive_patterns)]
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
