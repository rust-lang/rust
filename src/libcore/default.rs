// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The `Default` trait for types which may have meaningful default values

/// A trait that types which have a useful default value should implement.
pub trait Default {
    /// Return the "default value" for a type.
    fn default() -> Self;
}

macro_rules! default_impl(
    ($t:ty, $v:expr) => {
        impl Default for $t {
            #[inline]
            fn default() -> $t { $v }
        }
    }
)

default_impl!((), ())
default_impl!(bool, false)
default_impl!(char, '\x00')

default_impl!(uint, 0u)
default_impl!(u8,  0u8)
default_impl!(u16, 0u16)
default_impl!(u32, 0u32)
default_impl!(u64, 0u64)

default_impl!(int, 0i)
default_impl!(i8,  0i8)
default_impl!(i16, 0i16)
default_impl!(i32, 0i32)
default_impl!(i64, 0i64)

default_impl!(f32, 0.0f32)
default_impl!(f64, 0.0f64)

impl<T: Default + 'static> Default for @T {
    fn default() -> @T { @Default::default() }
}
