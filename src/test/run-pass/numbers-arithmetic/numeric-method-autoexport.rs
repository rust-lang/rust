// run-pass
// This file is intended to test only that methods are automatically
// reachable for each numeric type, for each exported impl, with no imports
// necessary. Testing the methods of the impls is done within the source
// file for each numeric type.

use std::ops::Add;

pub fn main() {
// ints
    // num
    assert_eq!(15_isize.add(6_isize), 21_isize);
    assert_eq!(15_i8.add(6i8), 21_i8);
    assert_eq!(15_i16.add(6i16), 21_i16);
    assert_eq!(15_i32.add(6i32), 21_i32);
    assert_eq!(15_i64.add(6i64), 21_i64);

// uints
    // num
    assert_eq!(15_usize.add(6_usize), 21_usize);
    assert_eq!(15_u8.add(6u8), 21_u8);
    assert_eq!(15_u16.add(6u16), 21_u16);
    assert_eq!(15_u32.add(6u32), 21_u32);
    assert_eq!(15_u64.add(6u64), 21_u64);
}
