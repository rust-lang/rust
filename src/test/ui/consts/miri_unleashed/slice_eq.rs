// compile-flags: -Zunleash-the-miri-inside-of-you
// run-pass

#![feature(const_raw_ptr_comparison)]

const EMPTY_SLICE: &[i32] = &[];
const EMPTY_EQ: bool = EMPTY_SLICE.as_ptr().guaranteed_eq(&[] as *const _);
const EMPTY_EQ2: bool = EMPTY_SLICE.as_ptr().guaranteed_ne(&[] as *const _);
const EMPTY_NE: bool = EMPTY_SLICE.as_ptr().guaranteed_ne(&[1] as *const _);
const EMPTY_NE2: bool = EMPTY_SLICE.as_ptr().guaranteed_eq(&[1] as *const _);

fn main() {
    assert!(!EMPTY_EQ);
    assert!(!EMPTY_EQ2);
    assert!(!EMPTY_NE);
    assert!(!EMPTY_NE2);
}
