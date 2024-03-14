//@ check-pass
//@ edition:2021
#![warn(unused_imports)]

use std::convert::TryFrom; //~ WARNING redundant import
use std::convert::TryInto; //~ WARNING redundant import

fn main() {
    let _e: Result<i32, _> = 8u8.try_into();
    let _f: Result<i32, _> = i32::try_from(8u8);
}
