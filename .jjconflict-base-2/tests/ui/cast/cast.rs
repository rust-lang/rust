//@ run-pass

#![allow(unused_assignments)]
#![allow(unused_variables)]

pub fn main() {
    let i: isize = 'Q' as isize;
    assert_eq!(i, 0x51);
    let u: u32 = i as u32;
    assert_eq!(u, 0x51 as u32);
    assert_eq!(u, 'Q' as u32);
    assert_eq!(i as u8, 'Q' as u8);
    assert_eq!(i as u8 as i8, 'Q' as u8 as i8);
    assert_eq!(0x51 as char, 'Q');
    assert_eq!(0 as u32, false as u32);

    // Test that `_` is correctly inferred.
    let x = &"hello";
    let mut y = x as *const _;
    y = core::ptr::null_mut();
}
