//@ check-pass
//@ edition:2021
#![warn(redundant_imports)]

use std::convert::TryFrom;//~ WARNING the item `TryFrom` is imported redundantly
use std::convert::TryInto;//~ WARNING the item `TryInto` is imported redundantly

fn main() {
    let _e: Result<i32, _> = 8u8.try_into();
    let _f: Result<i32, _> = i32::try_from(8u8);
}
