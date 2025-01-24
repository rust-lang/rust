//@ run-pass

#![feature(core_intrinsics)]

pub fn main() {
    use std::intrinsics::raw_eq;

    const RAW_EQ_I32_TRUE: bool = unsafe { raw_eq(&42_i32, &42) };
    assert!(RAW_EQ_I32_TRUE);

    const RAW_EQ_I32_FALSE: bool = unsafe { raw_eq(&4_i32, &2) };
    assert!(!RAW_EQ_I32_FALSE);

    const RAW_EQ_CHAR_TRUE: bool = unsafe { raw_eq(&'a', &'a') };
    assert!(RAW_EQ_CHAR_TRUE);

    const RAW_EQ_CHAR_FALSE: bool = unsafe { raw_eq(&'a', &'A') };
    assert!(!RAW_EQ_CHAR_FALSE);

    const RAW_EQ_ARRAY_TRUE: bool = unsafe { raw_eq(&[13_u8, 42], &[13, 42]) };
    assert!(RAW_EQ_ARRAY_TRUE);

    const RAW_EQ_ARRAY_FALSE: bool = unsafe { raw_eq(&[13_u8, 42], &[42, 13]) };
    assert!(!RAW_EQ_ARRAY_FALSE);
}
