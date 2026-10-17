//@ edition: 2015
//@ check-pass
#![warn(redundant_imports)]


use std::option::Option::Some;//~ WARNING the item `Some` is imported redundantly
use std::option::Option::None; //~ WARNING the item `None` is imported redundantly

use std::result::Result::Ok;//~ WARNING the item `Ok` is imported redundantly
use std::result::Result::Err;//~ WARNING the item `Err` is imported redundantly
use std::convert::{TryFrom, TryInto};

fn main() {
    let _a: Option<i32> = Some(1);
    let _b: Option<i32>  = None;
    let _c: Result<i32, String> = Ok(1);
    let _d: Result<i32, &str> = Err("error");
    let _e: Result<i32, _> = 8u8.try_into();
    let _f: Result<i32, _> = i32::try_from(8u8);
}
