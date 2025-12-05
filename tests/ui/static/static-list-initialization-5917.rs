// https://github.com/rust-lang/rust/issues/5917
//@ run-pass
#![allow(non_upper_case_globals)]

struct T (&'static [isize]);
static t : T = T (&[5, 4, 3]);
pub fn main () {
    let T(ref v) = t;
    assert_eq!(v[0], 5);
}
