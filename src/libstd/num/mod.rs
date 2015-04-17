// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Numeric traits and functions for generic mathematics
//!
//! These are implemented for the primitive numeric types in `std::{u8, u16,
//! u32, u64, usize, i8, i16, i32, i64, isize, f32, f64}`.

#![stable(feature = "rust1", since = "1.0.0")]
#![allow(missing_docs)]

#[cfg(test)] use fmt::Debug;

pub use core::num::{Zero, One};
pub use core::num::{FpCategory, ParseIntError, ParseFloatError};
pub use core::num::{wrapping, Wrapping};

/// Helper function for testing numeric operations
#[cfg(test)]
pub fn test_num<T>(ten: T, two: T) where
    T: PartialEq + NumCast
     + Add<Output=T> + Sub<Output=T>
     + Mul<Output=T> + Div<Output=T>
     + Rem<Output=T> + Debug
     + Copy
{
    assert_eq!(ten.add(two),  cast(12).unwrap());
    assert_eq!(ten.sub(two),  cast(8).unwrap());
    assert_eq!(ten.mul(two),  cast(20).unwrap());
    assert_eq!(ten.div(two),  cast(5).unwrap());
    assert_eq!(ten.rem(two),  cast(0).unwrap());

    assert_eq!(ten.add(two),  ten + two);
    assert_eq!(ten.sub(two),  ten - two);
    assert_eq!(ten.mul(two),  ten * two);
    assert_eq!(ten.div(two),  ten / two);
    assert_eq!(ten.rem(two),  ten % two);
}

#[cfg(test)]
mod tests {
    use core::prelude::*;
    use super::*;
    use i8;
    use i16;
    use i32;
    use i64;
    use isize;
    use u8;
    use u16;
    use u32;
    use u64;
    use usize;
    use string::ToString;

    macro_rules! test_cast_20 {
        ($_20:expr) => ({
            let _20 = $_20;

            assert_eq!(20usize, _20.to_uint().unwrap());
            assert_eq!(20u8,    _20.to_u8().unwrap());
            assert_eq!(20u16,   _20.to_u16().unwrap());
            assert_eq!(20u32,   _20.to_u32().unwrap());
            assert_eq!(20u64,   _20.to_u64().unwrap());
            assert_eq!(20,      _20.to_int().unwrap());
            assert_eq!(20i8,    _20.to_i8().unwrap());
            assert_eq!(20i16,   _20.to_i16().unwrap());
            assert_eq!(20i32,   _20.to_i32().unwrap());
            assert_eq!(20i64,   _20.to_i64().unwrap());
            assert_eq!(20f32,   _20.to_f32().unwrap());
            assert_eq!(20f64,   _20.to_f64().unwrap());

            assert_eq!(_20, NumCast::from(20usize).unwrap());
            assert_eq!(_20, NumCast::from(20u8).unwrap());
            assert_eq!(_20, NumCast::from(20u16).unwrap());
            assert_eq!(_20, NumCast::from(20u32).unwrap());
            assert_eq!(_20, NumCast::from(20u64).unwrap());
            assert_eq!(_20, NumCast::from(20).unwrap());
            assert_eq!(_20, NumCast::from(20i8).unwrap());
            assert_eq!(_20, NumCast::from(20i16).unwrap());
            assert_eq!(_20, NumCast::from(20i32).unwrap());
            assert_eq!(_20, NumCast::from(20i64).unwrap());
            assert_eq!(_20, NumCast::from(20f32).unwrap());
            assert_eq!(_20, NumCast::from(20f64).unwrap());

            assert_eq!(_20, cast(20usize).unwrap());
            assert_eq!(_20, cast(20u8).unwrap());
            assert_eq!(_20, cast(20u16).unwrap());
            assert_eq!(_20, cast(20u32).unwrap());
            assert_eq!(_20, cast(20u64).unwrap());
            assert_eq!(_20, cast(20).unwrap());
            assert_eq!(_20, cast(20i8).unwrap());
            assert_eq!(_20, cast(20i16).unwrap());
            assert_eq!(_20, cast(20i32).unwrap());
            assert_eq!(_20, cast(20i64).unwrap());
            assert_eq!(_20, cast(20f32).unwrap());
            assert_eq!(_20, cast(20f64).unwrap());
        })
    }

    #[test] fn test_u8_cast()    { test_cast_20!(20u8)    }
    #[test] fn test_u16_cast()   { test_cast_20!(20u16)   }
    #[test] fn test_u32_cast()   { test_cast_20!(20u32)   }
    #[test] fn test_u64_cast()   { test_cast_20!(20u64)   }
    #[test] fn test_uint_cast()  { test_cast_20!(20usize) }
    #[test] fn test_i8_cast()    { test_cast_20!(20i8)    }
    #[test] fn test_i16_cast()   { test_cast_20!(20i16)   }
    #[test] fn test_i32_cast()   { test_cast_20!(20i32)   }
    #[test] fn test_i64_cast()   { test_cast_20!(20i64)   }
    #[test] fn test_int_cast()   { test_cast_20!(20)      }
    #[test] fn test_f32_cast()   { test_cast_20!(20f32)   }
    #[test] fn test_f64_cast()   { test_cast_20!(20f64)   }

    #[test]
    fn test_cast_range_int_min() {
        assert_eq!(isize::MIN.to_int(),  Some(isize::MIN as isize));
        assert_eq!(isize::MIN.to_i8(),   None);
        assert_eq!(isize::MIN.to_i16(),  None);
        // isize::MIN.to_i32() is word-size specific
        assert_eq!(isize::MIN.to_i64(),  Some(isize::MIN as i64));
        assert_eq!(isize::MIN.to_uint(), None);
        assert_eq!(isize::MIN.to_u8(),   None);
        assert_eq!(isize::MIN.to_u16(),  None);
        assert_eq!(isize::MIN.to_u32(),  None);
        assert_eq!(isize::MIN.to_u64(),  None);

        #[cfg(target_pointer_width = "32")]
        fn check_word_size() {
            assert_eq!(isize::MIN.to_i32(), Some(isize::MIN as i32));
        }

        #[cfg(target_pointer_width = "64")]
        fn check_word_size() {
            assert_eq!(isize::MIN.to_i32(), None);
        }

        check_word_size();
    }

    #[test]
    fn test_cast_range_i8_min() {
        assert_eq!(i8::MIN.to_int(),  Some(i8::MIN as isize));
        assert_eq!(i8::MIN.to_i8(),   Some(i8::MIN as i8));
        assert_eq!(i8::MIN.to_i16(),  Some(i8::MIN as i16));
        assert_eq!(i8::MIN.to_i32(),  Some(i8::MIN as i32));
        assert_eq!(i8::MIN.to_i64(),  Some(i8::MIN as i64));
        assert_eq!(i8::MIN.to_uint(), None);
        assert_eq!(i8::MIN.to_u8(),   None);
        assert_eq!(i8::MIN.to_u16(),  None);
        assert_eq!(i8::MIN.to_u32(),  None);
        assert_eq!(i8::MIN.to_u64(),  None);
    }

    #[test]
    fn test_cast_range_i16_min() {
        assert_eq!(i16::MIN.to_int(),  Some(i16::MIN as isize));
        assert_eq!(i16::MIN.to_i8(),   None);
        assert_eq!(i16::MIN.to_i16(),  Some(i16::MIN as i16));
        assert_eq!(i16::MIN.to_i32(),  Some(i16::MIN as i32));
        assert_eq!(i16::MIN.to_i64(),  Some(i16::MIN as i64));
        assert_eq!(i16::MIN.to_uint(), None);
        assert_eq!(i16::MIN.to_u8(),   None);
        assert_eq!(i16::MIN.to_u16(),  None);
        assert_eq!(i16::MIN.to_u32(),  None);
        assert_eq!(i16::MIN.to_u64(),  None);
    }

    #[test]
    fn test_cast_range_i32_min() {
        assert_eq!(i32::MIN.to_int(),  Some(i32::MIN as isize));
        assert_eq!(i32::MIN.to_i8(),   None);
        assert_eq!(i32::MIN.to_i16(),  None);
        assert_eq!(i32::MIN.to_i32(),  Some(i32::MIN as i32));
        assert_eq!(i32::MIN.to_i64(),  Some(i32::MIN as i64));
        assert_eq!(i32::MIN.to_uint(), None);
        assert_eq!(i32::MIN.to_u8(),   None);
        assert_eq!(i32::MIN.to_u16(),  None);
        assert_eq!(i32::MIN.to_u32(),  None);
        assert_eq!(i32::MIN.to_u64(),  None);
    }

    #[test]
    fn test_cast_range_i64_min() {
        // i64::MIN.to_int() is word-size specific
        assert_eq!(i64::MIN.to_i8(),   None);
        assert_eq!(i64::MIN.to_i16(),  None);
        assert_eq!(i64::MIN.to_i32(),  None);
        assert_eq!(i64::MIN.to_i64(),  Some(i64::MIN as i64));
        assert_eq!(i64::MIN.to_uint(), None);
        assert_eq!(i64::MIN.to_u8(),   None);
        assert_eq!(i64::MIN.to_u16(),  None);
        assert_eq!(i64::MIN.to_u32(),  None);
        assert_eq!(i64::MIN.to_u64(),  None);

        #[cfg(target_pointer_width = "32")]
        fn check_word_size() {
            assert_eq!(i64::MIN.to_int(), None);
        }

        #[cfg(target_pointer_width = "64")]
        fn check_word_size() {
            assert_eq!(i64::MIN.to_int(), Some(i64::MIN as isize));
        }

        check_word_size();
    }

    #[test]
    fn test_cast_range_int_max() {
        assert_eq!(isize::MAX.to_int(),  Some(isize::MAX as isize));
        assert_eq!(isize::MAX.to_i8(),   None);
        assert_eq!(isize::MAX.to_i16(),  None);
        // isize::MAX.to_i32() is word-size specific
        assert_eq!(isize::MAX.to_i64(),  Some(isize::MAX as i64));
        assert_eq!(isize::MAX.to_u8(),   None);
        assert_eq!(isize::MAX.to_u16(),  None);
        // isize::MAX.to_u32() is word-size specific
        assert_eq!(isize::MAX.to_u64(),  Some(isize::MAX as u64));

        #[cfg(target_pointer_width = "32")]
        fn check_word_size() {
            assert_eq!(isize::MAX.to_i32(), Some(isize::MAX as i32));
            assert_eq!(isize::MAX.to_u32(), Some(isize::MAX as u32));
        }

        #[cfg(target_pointer_width = "64")]
        fn check_word_size() {
            assert_eq!(isize::MAX.to_i32(), None);
            assert_eq!(isize::MAX.to_u32(), None);
        }

        check_word_size();
    }

    #[test]
    fn test_cast_range_i8_max() {
        assert_eq!(i8::MAX.to_int(),  Some(i8::MAX as isize));
        assert_eq!(i8::MAX.to_i8(),   Some(i8::MAX as i8));
        assert_eq!(i8::MAX.to_i16(),  Some(i8::MAX as i16));
        assert_eq!(i8::MAX.to_i32(),  Some(i8::MAX as i32));
        assert_eq!(i8::MAX.to_i64(),  Some(i8::MAX as i64));
        assert_eq!(i8::MAX.to_uint(), Some(i8::MAX as usize));
        assert_eq!(i8::MAX.to_u8(),   Some(i8::MAX as u8));
        assert_eq!(i8::MAX.to_u16(),  Some(i8::MAX as u16));
        assert_eq!(i8::MAX.to_u32(),  Some(i8::MAX as u32));
        assert_eq!(i8::MAX.to_u64(),  Some(i8::MAX as u64));
    }

    #[test]
    fn test_cast_range_i16_max() {
        assert_eq!(i16::MAX.to_int(),  Some(i16::MAX as isize));
        assert_eq!(i16::MAX.to_i8(),   None);
        assert_eq!(i16::MAX.to_i16(),  Some(i16::MAX as i16));
        assert_eq!(i16::MAX.to_i32(),  Some(i16::MAX as i32));
        assert_eq!(i16::MAX.to_i64(),  Some(i16::MAX as i64));
        assert_eq!(i16::MAX.to_uint(), Some(i16::MAX as usize));
        assert_eq!(i16::MAX.to_u8(),   None);
        assert_eq!(i16::MAX.to_u16(),  Some(i16::MAX as u16));
        assert_eq!(i16::MAX.to_u32(),  Some(i16::MAX as u32));
        assert_eq!(i16::MAX.to_u64(),  Some(i16::MAX as u64));
    }

    #[test]
    fn test_cast_range_i32_max() {
        assert_eq!(i32::MAX.to_int(),  Some(i32::MAX as isize));
        assert_eq!(i32::MAX.to_i8(),   None);
        assert_eq!(i32::MAX.to_i16(),  None);
        assert_eq!(i32::MAX.to_i32(),  Some(i32::MAX as i32));
        assert_eq!(i32::MAX.to_i64(),  Some(i32::MAX as i64));
        assert_eq!(i32::MAX.to_uint(), Some(i32::MAX as usize));
        assert_eq!(i32::MAX.to_u8(),   None);
        assert_eq!(i32::MAX.to_u16(),  None);
        assert_eq!(i32::MAX.to_u32(),  Some(i32::MAX as u32));
        assert_eq!(i32::MAX.to_u64(),  Some(i32::MAX as u64));
    }

    #[test]
    fn test_cast_range_i64_max() {
        // i64::MAX.to_int() is word-size specific
        assert_eq!(i64::MAX.to_i8(),   None);
        assert_eq!(i64::MAX.to_i16(),  None);
        assert_eq!(i64::MAX.to_i32(),  None);
        assert_eq!(i64::MAX.to_i64(),  Some(i64::MAX as i64));
        // i64::MAX.to_uint() is word-size specific
        assert_eq!(i64::MAX.to_u8(),   None);
        assert_eq!(i64::MAX.to_u16(),  None);
        assert_eq!(i64::MAX.to_u32(),  None);
        assert_eq!(i64::MAX.to_u64(),  Some(i64::MAX as u64));

        #[cfg(target_pointer_width = "32")]
        fn check_word_size() {
            assert_eq!(i64::MAX.to_int(),  None);
            assert_eq!(i64::MAX.to_uint(), None);
        }

        #[cfg(target_pointer_width = "64")]
        fn check_word_size() {
            assert_eq!(i64::MAX.to_int(),  Some(i64::MAX as isize));
            assert_eq!(i64::MAX.to_uint(), Some(i64::MAX as usize));
        }

        check_word_size();
    }

    #[test]
    fn test_cast_range_uint_min() {
        assert_eq!(usize::MIN.to_int(),  Some(usize::MIN as isize));
        assert_eq!(usize::MIN.to_i8(),   Some(usize::MIN as i8));
        assert_eq!(usize::MIN.to_i16(),  Some(usize::MIN as i16));
        assert_eq!(usize::MIN.to_i32(),  Some(usize::MIN as i32));
        assert_eq!(usize::MIN.to_i64(),  Some(usize::MIN as i64));
        assert_eq!(usize::MIN.to_uint(), Some(usize::MIN as usize));
        assert_eq!(usize::MIN.to_u8(),   Some(usize::MIN as u8));
        assert_eq!(usize::MIN.to_u16(),  Some(usize::MIN as u16));
        assert_eq!(usize::MIN.to_u32(),  Some(usize::MIN as u32));
        assert_eq!(usize::MIN.to_u64(),  Some(usize::MIN as u64));
    }

    #[test]
    fn test_cast_range_u8_min() {
        assert_eq!(u8::MIN.to_int(),  Some(u8::MIN as isize));
        assert_eq!(u8::MIN.to_i8(),   Some(u8::MIN as i8));
        assert_eq!(u8::MIN.to_i16(),  Some(u8::MIN as i16));
        assert_eq!(u8::MIN.to_i32(),  Some(u8::MIN as i32));
        assert_eq!(u8::MIN.to_i64(),  Some(u8::MIN as i64));
        assert_eq!(u8::MIN.to_uint(), Some(u8::MIN as usize));
        assert_eq!(u8::MIN.to_u8(),   Some(u8::MIN as u8));
        assert_eq!(u8::MIN.to_u16(),  Some(u8::MIN as u16));
        assert_eq!(u8::MIN.to_u32(),  Some(u8::MIN as u32));
        assert_eq!(u8::MIN.to_u64(),  Some(u8::MIN as u64));
    }

    #[test]
    fn test_cast_range_u16_min() {
        assert_eq!(u16::MIN.to_int(),  Some(u16::MIN as isize));
        assert_eq!(u16::MIN.to_i8(),   Some(u16::MIN as i8));
        assert_eq!(u16::MIN.to_i16(),  Some(u16::MIN as i16));
        assert_eq!(u16::MIN.to_i32(),  Some(u16::MIN as i32));
        assert_eq!(u16::MIN.to_i64(),  Some(u16::MIN as i64));
        assert_eq!(u16::MIN.to_uint(), Some(u16::MIN as usize));
        assert_eq!(u16::MIN.to_u8(),   Some(u16::MIN as u8));
        assert_eq!(u16::MIN.to_u16(),  Some(u16::MIN as u16));
        assert_eq!(u16::MIN.to_u32(),  Some(u16::MIN as u32));
        assert_eq!(u16::MIN.to_u64(),  Some(u16::MIN as u64));
    }

    #[test]
    fn test_cast_range_u32_min() {
        assert_eq!(u32::MIN.to_int(),  Some(u32::MIN as isize));
        assert_eq!(u32::MIN.to_i8(),   Some(u32::MIN as i8));
        assert_eq!(u32::MIN.to_i16(),  Some(u32::MIN as i16));
        assert_eq!(u32::MIN.to_i32(),  Some(u32::MIN as i32));
        assert_eq!(u32::MIN.to_i64(),  Some(u32::MIN as i64));
        assert_eq!(u32::MIN.to_uint(), Some(u32::MIN as usize));
        assert_eq!(u32::MIN.to_u8(),   Some(u32::MIN as u8));
        assert_eq!(u32::MIN.to_u16(),  Some(u32::MIN as u16));
        assert_eq!(u32::MIN.to_u32(),  Some(u32::MIN as u32));
        assert_eq!(u32::MIN.to_u64(),  Some(u32::MIN as u64));
    }

    #[test]
    fn test_cast_range_u64_min() {
        assert_eq!(u64::MIN.to_int(),  Some(u64::MIN as isize));
        assert_eq!(u64::MIN.to_i8(),   Some(u64::MIN as i8));
        assert_eq!(u64::MIN.to_i16(),  Some(u64::MIN as i16));
        assert_eq!(u64::MIN.to_i32(),  Some(u64::MIN as i32));
        assert_eq!(u64::MIN.to_i64(),  Some(u64::MIN as i64));
        assert_eq!(u64::MIN.to_uint(), Some(u64::MIN as usize));
        assert_eq!(u64::MIN.to_u8(),   Some(u64::MIN as u8));
        assert_eq!(u64::MIN.to_u16(),  Some(u64::MIN as u16));
        assert_eq!(u64::MIN.to_u32(),  Some(u64::MIN as u32));
        assert_eq!(u64::MIN.to_u64(),  Some(u64::MIN as u64));
    }

    #[test]
    fn test_cast_range_uint_max() {
        assert_eq!(usize::MAX.to_int(),  None);
        assert_eq!(usize::MAX.to_i8(),   None);
        assert_eq!(usize::MAX.to_i16(),  None);
        assert_eq!(usize::MAX.to_i32(),  None);
        // usize::MAX.to_i64() is word-size specific
        assert_eq!(usize::MAX.to_u8(),   None);
        assert_eq!(usize::MAX.to_u16(),  None);
        // usize::MAX.to_u32() is word-size specific
        assert_eq!(usize::MAX.to_u64(),  Some(usize::MAX as u64));

        #[cfg(target_pointer_width = "32")]
        fn check_word_size() {
            assert_eq!(usize::MAX.to_u32(), Some(usize::MAX as u32));
            assert_eq!(usize::MAX.to_i64(), Some(usize::MAX as i64));
        }

        #[cfg(target_pointer_width = "64")]
        fn check_word_size() {
            assert_eq!(usize::MAX.to_u32(), None);
            assert_eq!(usize::MAX.to_i64(), None);
        }

        check_word_size();
    }

    #[test]
    fn test_cast_range_u8_max() {
        assert_eq!(u8::MAX.to_int(),  Some(u8::MAX as isize));
        assert_eq!(u8::MAX.to_i8(),   None);
        assert_eq!(u8::MAX.to_i16(),  Some(u8::MAX as i16));
        assert_eq!(u8::MAX.to_i32(),  Some(u8::MAX as i32));
        assert_eq!(u8::MAX.to_i64(),  Some(u8::MAX as i64));
        assert_eq!(u8::MAX.to_uint(), Some(u8::MAX as usize));
        assert_eq!(u8::MAX.to_u8(),   Some(u8::MAX as u8));
        assert_eq!(u8::MAX.to_u16(),  Some(u8::MAX as u16));
        assert_eq!(u8::MAX.to_u32(),  Some(u8::MAX as u32));
        assert_eq!(u8::MAX.to_u64(),  Some(u8::MAX as u64));
    }

    #[test]
    fn test_cast_range_u16_max() {
        assert_eq!(u16::MAX.to_int(),  Some(u16::MAX as isize));
        assert_eq!(u16::MAX.to_i8(),   None);
        assert_eq!(u16::MAX.to_i16(),  None);
        assert_eq!(u16::MAX.to_i32(),  Some(u16::MAX as i32));
        assert_eq!(u16::MAX.to_i64(),  Some(u16::MAX as i64));
        assert_eq!(u16::MAX.to_uint(), Some(u16::MAX as usize));
        assert_eq!(u16::MAX.to_u8(),   None);
        assert_eq!(u16::MAX.to_u16(),  Some(u16::MAX as u16));
        assert_eq!(u16::MAX.to_u32(),  Some(u16::MAX as u32));
        assert_eq!(u16::MAX.to_u64(),  Some(u16::MAX as u64));
    }

    #[test]
    fn test_cast_range_u32_max() {
        // u32::MAX.to_int() is word-size specific
        assert_eq!(u32::MAX.to_i8(),   None);
        assert_eq!(u32::MAX.to_i16(),  None);
        assert_eq!(u32::MAX.to_i32(),  None);
        assert_eq!(u32::MAX.to_i64(),  Some(u32::MAX as i64));
        assert_eq!(u32::MAX.to_uint(), Some(u32::MAX as usize));
        assert_eq!(u32::MAX.to_u8(),   None);
        assert_eq!(u32::MAX.to_u16(),  None);
        assert_eq!(u32::MAX.to_u32(),  Some(u32::MAX as u32));
        assert_eq!(u32::MAX.to_u64(),  Some(u32::MAX as u64));

        #[cfg(target_pointer_width = "32")]
        fn check_word_size() {
            assert_eq!(u32::MAX.to_int(),  None);
        }

        #[cfg(target_pointer_width = "64")]
        fn check_word_size() {
            assert_eq!(u32::MAX.to_int(),  Some(u32::MAX as isize));
        }

        check_word_size();
    }

    #[test]
    fn test_cast_range_u64_max() {
        assert_eq!(u64::MAX.to_int(),  None);
        assert_eq!(u64::MAX.to_i8(),   None);
        assert_eq!(u64::MAX.to_i16(),  None);
        assert_eq!(u64::MAX.to_i32(),  None);
        assert_eq!(u64::MAX.to_i64(),  None);
        // u64::MAX.to_uint() is word-size specific
        assert_eq!(u64::MAX.to_u8(),   None);
        assert_eq!(u64::MAX.to_u16(),  None);
        assert_eq!(u64::MAX.to_u32(),  None);
        assert_eq!(u64::MAX.to_u64(),  Some(u64::MAX as u64));

        #[cfg(target_pointer_width = "32")]
        fn check_word_size() {
            assert_eq!(u64::MAX.to_uint(), None);
        }

        #[cfg(target_pointer_width = "64")]
        fn check_word_size() {
            assert_eq!(u64::MAX.to_uint(), Some(u64::MAX as usize));
        }

        check_word_size();
    }

    #[test]
    fn test_saturating_add_uint() {
        use usize::MAX;
        assert_eq!(3_usize.saturating_add(5_usize), 8_usize);
        assert_eq!(3_usize.saturating_add(MAX-1), MAX);
        assert_eq!(MAX.saturating_add(MAX), MAX);
        assert_eq!((MAX-2).saturating_add(1), MAX-1);
    }

    #[test]
    fn test_saturating_sub_uint() {
        use usize::MAX;
        assert_eq!(5_usize.saturating_sub(3_usize), 2_usize);
        assert_eq!(3_usize.saturating_sub(5_usize), 0_usize);
        assert_eq!(0_usize.saturating_sub(1_usize), 0_usize);
        assert_eq!((MAX-1).saturating_sub(MAX), 0);
    }

    #[test]
    fn test_saturating_add_int() {
        use isize::{MIN,MAX};
        assert_eq!(3.saturating_add(5), 8);
        assert_eq!(3.saturating_add(MAX-1), MAX);
        assert_eq!(MAX.saturating_add(MAX), MAX);
        assert_eq!((MAX-2).saturating_add(1), MAX-1);
        assert_eq!(3.saturating_add(-5), -2);
        assert_eq!(MIN.saturating_add(-1), MIN);
        assert_eq!((-2).saturating_add(-MAX), MIN);
    }

    #[test]
    fn test_saturating_sub_int() {
        use isize::{MIN,MAX};
        assert_eq!(3.saturating_sub(5), -2);
        assert_eq!(MIN.saturating_sub(1), MIN);
        assert_eq!((-2).saturating_sub(MAX), MIN);
        assert_eq!(3.saturating_sub(-5), 8);
        assert_eq!(3.saturating_sub(-(MAX-1)), MAX);
        assert_eq!(MAX.saturating_sub(-MAX), MAX);
        assert_eq!((MAX-2).saturating_sub(-1), MAX-1);
    }

    #[test]
    fn test_checked_add() {
        let five_less = usize::MAX - 5;
        assert_eq!(five_less.checked_add(0), Some(usize::MAX - 5));
        assert_eq!(five_less.checked_add(1), Some(usize::MAX - 4));
        assert_eq!(five_less.checked_add(2), Some(usize::MAX - 3));
        assert_eq!(five_less.checked_add(3), Some(usize::MAX - 2));
        assert_eq!(five_less.checked_add(4), Some(usize::MAX - 1));
        assert_eq!(five_less.checked_add(5), Some(usize::MAX));
        assert_eq!(five_less.checked_add(6), None);
        assert_eq!(five_less.checked_add(7), None);
    }

    #[test]
    fn test_checked_sub() {
        assert_eq!(5_usize.checked_sub(0), Some(5));
        assert_eq!(5_usize.checked_sub(1), Some(4));
        assert_eq!(5_usize.checked_sub(2), Some(3));
        assert_eq!(5_usize.checked_sub(3), Some(2));
        assert_eq!(5_usize.checked_sub(4), Some(1));
        assert_eq!(5_usize.checked_sub(5), Some(0));
        assert_eq!(5_usize.checked_sub(6), None);
        assert_eq!(5_usize.checked_sub(7), None);
    }

    #[test]
    fn test_checked_mul() {
        let third = usize::MAX / 3;
        assert_eq!(third.checked_mul(0), Some(0));
        assert_eq!(third.checked_mul(1), Some(third));
        assert_eq!(third.checked_mul(2), Some(third * 2));
        assert_eq!(third.checked_mul(3), Some(third * 3));
        assert_eq!(third.checked_mul(4), None);
    }

    macro_rules! test_is_power_of_two {
        ($test_name:ident, $T:ident) => (
            fn $test_name() {
                #![test]
                assert_eq!((0 as $T).is_power_of_two(), false);
                assert_eq!((1 as $T).is_power_of_two(), true);
                assert_eq!((2 as $T).is_power_of_two(), true);
                assert_eq!((3 as $T).is_power_of_two(), false);
                assert_eq!((4 as $T).is_power_of_two(), true);
                assert_eq!((5 as $T).is_power_of_two(), false);
                assert_eq!(($T::MAX / 2 + 1).is_power_of_two(), true);
            }
        )
    }

    test_is_power_of_two!{ test_is_power_of_two_u8, u8 }
    test_is_power_of_two!{ test_is_power_of_two_u16, u16 }
    test_is_power_of_two!{ test_is_power_of_two_u32, u32 }
    test_is_power_of_two!{ test_is_power_of_two_u64, u64 }
    test_is_power_of_two!{ test_is_power_of_two_uint, usize }

    macro_rules! test_next_power_of_two {
        ($test_name:ident, $T:ident) => (
            fn $test_name() {
                #![test]
                assert_eq!((0 as $T).next_power_of_two(), 1);
                let mut next_power = 1;
                for i in 1 as $T..40 {
                     assert_eq!(i.next_power_of_two(), next_power);
                     if i == next_power { next_power *= 2 }
                }
            }
        )
    }

    test_next_power_of_two! { test_next_power_of_two_u8, u8 }
    test_next_power_of_two! { test_next_power_of_two_u16, u16 }
    test_next_power_of_two! { test_next_power_of_two_u32, u32 }
    test_next_power_of_two! { test_next_power_of_two_u64, u64 }
    test_next_power_of_two! { test_next_power_of_two_uint, usize }

    macro_rules! test_checked_next_power_of_two {
        ($test_name:ident, $T:ident) => (
            fn $test_name() {
                #![test]
                assert_eq!((0 as $T).checked_next_power_of_two(), Some(1));
                assert!(($T::MAX / 2).checked_next_power_of_two().is_some());
                assert_eq!(($T::MAX - 1).checked_next_power_of_two(), None);
                assert_eq!($T::MAX.checked_next_power_of_two(), None);
                let mut next_power = 1;
                for i in 1 as $T..40 {
                     assert_eq!(i.checked_next_power_of_two(), Some(next_power));
                     if i == next_power { next_power *= 2 }
                }
            }
        )
    }

    test_checked_next_power_of_two! { test_checked_next_power_of_two_u8, u8 }
    test_checked_next_power_of_two! { test_checked_next_power_of_two_u16, u16 }
    test_checked_next_power_of_two! { test_checked_next_power_of_two_u32, u32 }
    test_checked_next_power_of_two! { test_checked_next_power_of_two_u64, u64 }
    test_checked_next_power_of_two! { test_checked_next_power_of_two_uint, usize }

    #[derive(PartialEq, Debug)]
    struct Value { x: isize }

    impl ToPrimitive for Value {
        fn to_i64(&self) -> Option<i64> { self.x.to_i64() }
        fn to_u64(&self) -> Option<u64> { self.x.to_u64() }
    }

    impl FromPrimitive for Value {
        fn from_i64(n: i64) -> Option<Value> { Some(Value { x: n as isize }) }
        fn from_u64(n: u64) -> Option<Value> { Some(Value { x: n as isize }) }
    }

    #[test]
    fn test_to_primitive() {
        let value = Value { x: 5 };
        assert_eq!(value.to_int(),  Some(5));
        assert_eq!(value.to_i8(),   Some(5));
        assert_eq!(value.to_i16(),  Some(5));
        assert_eq!(value.to_i32(),  Some(5));
        assert_eq!(value.to_i64(),  Some(5));
        assert_eq!(value.to_uint(), Some(5));
        assert_eq!(value.to_u8(),   Some(5));
        assert_eq!(value.to_u16(),  Some(5));
        assert_eq!(value.to_u32(),  Some(5));
        assert_eq!(value.to_u64(),  Some(5));
        assert_eq!(value.to_f32(),  Some(5f32));
        assert_eq!(value.to_f64(),  Some(5f64));
    }

    #[test]
    fn test_from_primitive() {
        assert_eq!(from_int(5),    Some(Value { x: 5 }));
        assert_eq!(from_i8(5),     Some(Value { x: 5 }));
        assert_eq!(from_i16(5),    Some(Value { x: 5 }));
        assert_eq!(from_i32(5),    Some(Value { x: 5 }));
        assert_eq!(from_i64(5),    Some(Value { x: 5 }));
        assert_eq!(from_uint(5),   Some(Value { x: 5 }));
        assert_eq!(from_u8(5),     Some(Value { x: 5 }));
        assert_eq!(from_u16(5),    Some(Value { x: 5 }));
        assert_eq!(from_u32(5),    Some(Value { x: 5 }));
        assert_eq!(from_u64(5),    Some(Value { x: 5 }));
        assert_eq!(from_f32(5f32), Some(Value { x: 5 }));
        assert_eq!(from_f64(5f64), Some(Value { x: 5 }));
    }

    #[test]
    fn test_pow() {
        fn naive_pow<T: Int>(base: T, exp: usize) -> T {
            let one: T = Int::one();
            (0..exp).fold(one, |acc, _| acc * base)
        }
        macro_rules! assert_pow {
            (($num:expr, $exp:expr) => $expected:expr) => {{
                let result = $num.pow($exp);
                assert_eq!(result, $expected);
                assert_eq!(result, naive_pow($num, $exp));
            }}
        }
        assert_pow!((3,     0 ) => 1);
        assert_pow!((5,     1 ) => 5);
        assert_pow!((-4,    2 ) => 16);
        assert_pow!((8,     3 ) => 512);
        assert_pow!((2u64,   50) => 1125899906842624);
    }

    #[test]
    fn test_uint_to_str_overflow() {
        let mut u8_val: u8 = 255;
        assert_eq!(u8_val.to_string(), "255");

        u8_val = u8_val.wrapping_add(1);
        assert_eq!(u8_val.to_string(), "0");

        let mut u16_val: u16 = 65_535;
        assert_eq!(u16_val.to_string(), "65535");

        u16_val = u16_val.wrapping_add(1);
        assert_eq!(u16_val.to_string(), "0");

        let mut u32_val: u32 = 4_294_967_295;
        assert_eq!(u32_val.to_string(), "4294967295");

        u32_val = u32_val.wrapping_add(1);
        assert_eq!(u32_val.to_string(), "0");

        let mut u64_val: u64 = 18_446_744_073_709_551_615;
        assert_eq!(u64_val.to_string(), "18446744073709551615");

        u64_val = u64_val.wrapping_add(1);
        assert_eq!(u64_val.to_string(), "0");
    }

    fn from_str<T: ::str::FromStr>(t: &str) -> Option<T> {
        ::str::FromStr::from_str(t).ok()
    }

    #[test]
    fn test_uint_from_str_overflow() {
        let mut u8_val: u8 = 255;
        assert_eq!(from_str::<u8>("255"), Some(u8_val));
        assert_eq!(from_str::<u8>("256"), None);

        u8_val = u8_val.wrapping_add(1);
        assert_eq!(from_str::<u8>("0"), Some(u8_val));
        assert_eq!(from_str::<u8>("-1"), None);

        let mut u16_val: u16 = 65_535;
        assert_eq!(from_str::<u16>("65535"), Some(u16_val));
        assert_eq!(from_str::<u16>("65536"), None);

        u16_val = u16_val.wrapping_add(1);
        assert_eq!(from_str::<u16>("0"), Some(u16_val));
        assert_eq!(from_str::<u16>("-1"), None);

        let mut u32_val: u32 = 4_294_967_295;
        assert_eq!(from_str::<u32>("4294967295"), Some(u32_val));
        assert_eq!(from_str::<u32>("4294967296"), None);

        u32_val = u32_val.wrapping_add(1);
        assert_eq!(from_str::<u32>("0"), Some(u32_val));
        assert_eq!(from_str::<u32>("-1"), None);

        let mut u64_val: u64 = 18_446_744_073_709_551_615;
        assert_eq!(from_str::<u64>("18446744073709551615"), Some(u64_val));
        assert_eq!(from_str::<u64>("18446744073709551616"), None);

        u64_val = u64_val.wrapping_add(1);
        assert_eq!(from_str::<u64>("0"), Some(u64_val));
        assert_eq!(from_str::<u64>("-1"), None);
    }
}


#[cfg(test)]
mod bench {
    extern crate test;
    use self::test::Bencher;
    use num::Int;
    use prelude::v1::*;

    #[bench]
    fn bench_pow_function(b: &mut Bencher) {
        let v = (0..1024).collect::<Vec<_>>();
        b.iter(|| {v.iter().fold(0, |old, new| old.pow(*new as u32));});
    }
}
