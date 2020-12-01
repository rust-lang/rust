// run-pass
#![allow(overflowing_literals)]

#[deny(const_err)]

fn main() {
    #[cfg(target_pointer_width = "32")]
    const I: isize = -2147483648isize;
    #[cfg(target_pointer_width = "64")]
    const I: isize = -9223372036854775808isize;
    assert_eq!(i32::MIN as u64, 0xffffffff80000000);
    assert_eq!(-2147483648isize as u64, 0xffffffff80000000);
    assert_eq!(-2147483648i32 as u64, 0xffffffff80000000);
    assert_eq!(i64::MIN as u64, 0x8000000000000000);
    #[cfg(target_pointer_width = "64")]
    assert_eq!(-9223372036854775808isize as u64, 0x8000000000000000);
    #[cfg(target_pointer_width = "32")]
    assert_eq!(-9223372036854775808isize as u64, 0);
    assert_eq!(-9223372036854775808i32 as u64, 0);
    const J: usize = i32::MAX as usize;
    const K: usize = -1i32 as u32 as usize;
    const L: usize = i32::MIN as usize;
    const M: usize = i64::MIN as usize;
    match 5 {
        J => {},
        K => {},
        L => {},
        M => {},
        _ => {}
    }
    match 5 {
        I => {},
        _ => {}
    }
}
