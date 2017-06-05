// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use fmt;

/// Internal trait for APIs like `i32::cast`.
#[unstable(feature = "num_cast_internals", issue = "0")]
pub trait Cast<T>: Sized {
    /// Internal implementation detail of this trait.
    fn cast(t: T) -> Result<Self, CastError>;
}

/// Error type returned from APIs like `i32::cast`, indicates that a cast could
/// not be performed losslessly.
#[unstable(feature = "num_cast", issue = "0")]
#[derive(Debug)]
pub struct CastError(());

#[unstable(feature = "num_cast", issue = "0")]
impl fmt::Display for CastError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        "failed to losslessly cast integral types".fmt(f)
    }
}

macro_rules! same_sign_cast_int_impl {
    ($storage:ty, $target:ty, $($source:ty),*) => {$(
        #[unstable(feature = "num_cast", issue = "0")]
        impl Cast<$source> for $target {
            #[inline]
            fn cast(u: $source) -> Result<$target, CastError> {
                let min = <$target>::min_value() as $storage;
                let max = <$target>::max_value() as $storage;
                if u as $storage < min || u as $storage > max {
                    Err(CastError(()))
                } else {
                    Ok(u as $target)
                }
            }
        }
    )*}
}

same_sign_cast_int_impl!(u128, u8, u8, u16, u32, u64, u128, usize);
same_sign_cast_int_impl!(i128, i8, i8, i16, i32, i64, i128, isize);
same_sign_cast_int_impl!(u128, u16, u8, u16, u32, u64, u128, usize);
same_sign_cast_int_impl!(i128, i16, i8, i16, i32, i64, i128, isize);
same_sign_cast_int_impl!(u128, u32, u8, u16, u32, u64, u128, usize);
same_sign_cast_int_impl!(i128, i32, i8, i16, i32, i64, i128, isize);
same_sign_cast_int_impl!(u128, u64, u8, u16, u32, u64, u128, usize);
same_sign_cast_int_impl!(i128, i64, i8, i16, i32, i64, i128, isize);
same_sign_cast_int_impl!(u128, u128, u8, u16, u32, u64, u128, usize);
same_sign_cast_int_impl!(i128, i128, i8, i16, i32, i64, i128, isize);
same_sign_cast_int_impl!(u128, usize, u8, u16, u32, u64, u128, usize);
same_sign_cast_int_impl!(i128, isize, i8, i16, i32, i64, i128, isize);

macro_rules! cross_sign_cast_int_impl {
    ($unsigned:ty, $($signed:ty),*) => {$(
        #[unstable(feature = "num_cast", issue = "0")]
        impl Cast<$unsigned> for $signed {
            #[inline]
            fn cast(u: $unsigned) -> Result<$signed, CastError> {
                let max = <$signed>::max_value() as u128;
                if u as u128 > max {
                    Err(CastError(()))
                } else {
                    Ok(u as $signed)
                }
            }
        }

        #[unstable(feature = "num_cast", issue = "0")]
        impl Cast<$signed> for $unsigned {
            #[inline]
            fn cast(u: $signed) -> Result<$unsigned, CastError> {
                let max = <$unsigned>::max_value() as u128;
                if u < 0 || u as u128 > max {
                    Err(CastError(()))
                } else {
                    Ok(u as $unsigned)
                }
            }
        }
    )*}
}

cross_sign_cast_int_impl!(u8, i8, i16, i32, i64, i128, isize);
cross_sign_cast_int_impl!(u16, i8, i16, i32, i64, i128, isize);
cross_sign_cast_int_impl!(u32, i8, i16, i32, i64, i128, isize);
cross_sign_cast_int_impl!(u64, i8, i16, i32, i64, i128, isize);
cross_sign_cast_int_impl!(u128, i8, i16, i32, i64, i128, isize);
cross_sign_cast_int_impl!(usize, i8, i16, i32, i64, i128, isize);
