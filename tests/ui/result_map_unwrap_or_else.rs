// Copyright 2014-2019 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Checks implementation of `RESULT_MAP_UNWRAP_OR_ELSE`

#![warn(clippy::result_map_unwrap_or_else)]

include!("../auxiliary/option_helpers.rs");

fn result_methods() {
    let res: Result<i32, ()> = Ok(1);

    // Check RESULT_MAP_UNWRAP_OR_ELSE
    // single line case
    let _ = res.map(|x| x + 1).unwrap_or_else(|e| 0); // should lint even though this call is on a separate line
                                                      // multi line cases
    let _ = res.map(|x| x + 1).unwrap_or_else(|e| 0);
    let _ = res.map(|x| x + 1).unwrap_or_else(|e| 0);
    // macro case
    let _ = opt_map!(res, |x| x + 1).unwrap_or_else(|e| 0); // should not lint
}

fn main() {}
