#![feature(plugin)]
#![plugin(clippy)]
#![allow(unused_imports)]
#![deny(unsafe_removed_from_name)]

use std::cell::{UnsafeCell as TotallySafeCell};
//~^ ERROR removed "unsafe" from the name of `UnsafeCell` in use as `TotallySafeCell`

use std::cell::UnsafeCell as TotallySafeCellAgain;
//~^ ERROR removed "unsafe" from the name of `UnsafeCell` in use as `TotallySafeCellAgain`

fn main() {}
