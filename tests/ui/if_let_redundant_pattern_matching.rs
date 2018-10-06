// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


#![feature(tool_lints)]


#![warn(clippy::all)]
#![warn(clippy::if_let_redundant_pattern_matching)]


fn main() {
    if let Ok(_) = Ok::<i32, i32>(42) {}

    if let Err(_) = Err::<i32, i32>(42) {
    }

    if let None = None::<()> {
    }

    if let Some(_) = Some(42) {
    }

    if Ok::<i32, i32>(42).is_ok() {
    }

    if Err::<i32, i32>(42).is_err() {
    }

    if None::<i32>.is_none() {
    }

    if Some(42).is_some() {
    }

    if let Ok(x) = Ok::<i32,i32>(42) {
        println!("{}", x);
    }
}
