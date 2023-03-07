// compile-flags: -Zunleash-the-miri-inside-of-you
// run-pass

#![feature(const_raw_ptr_comparison)]

const EMPTY_SLICE: &[i32] = &[];
const EMPTY_EQ: Option<bool> = EMPTY_SLICE.as_ptr().guaranteed_eq(&[] as *const _);
const EMPTY_EQ2: Option<bool> = EMPTY_SLICE.as_ptr().guaranteed_eq(&[1] as *const _);

fn main() {
    assert!(EMPTY_EQ.is_none());
    assert!(EMPTY_EQ2.is_none());
}
