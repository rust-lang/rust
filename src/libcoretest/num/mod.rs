// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::num::cast;

mod int_macros;
mod i8;
mod i16;
mod i32;
mod i64;
mod int;
mod uint_macros;
mod u8;
mod u16;
mod u32;
mod u64;
mod uint;

/// Helper function for testing numeric operations
pub fn test_num<T:Num + NumCast + ::std::fmt::Show>(ten: T, two: T) {
    assert_eq!(ten.add(&two),  cast(12i).unwrap());
    assert_eq!(ten.sub(&two),  cast(8i).unwrap());
    assert_eq!(ten.mul(&two),  cast(20i).unwrap());
    assert_eq!(ten.div(&two),  cast(5i).unwrap());
    assert_eq!(ten.rem(&two),  cast(0i).unwrap());

    assert_eq!(ten.add(&two),  ten + two);
    assert_eq!(ten.sub(&two),  ten - two);
    assert_eq!(ten.mul(&two),  ten * two);
    assert_eq!(ten.div(&two),  ten / two);
    assert_eq!(ten.rem(&two),  ten % two);
}
