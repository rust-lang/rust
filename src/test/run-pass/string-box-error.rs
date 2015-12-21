// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Ensure that both `Box<Error + Send + Sync>` and `Box<Error>` can be obtained from `String`.

use std::error::Error;

fn main() {
    let _err1: Box<Error + Send + Sync> = From::from("test".to_string());
    let _err2: Box<Error> = From::from("test".to_string());
}
