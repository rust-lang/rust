// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:static-methods-crate.rs

extern crate static_methods_crate;

use static_methods_crate::read;

pub fn main() {
    let result: int = read("5".to_owned());
    assert_eq!(result, 5);
    assert_eq!(read::readMaybe("false".to_owned()), Some(false));
    assert_eq!(read::readMaybe("foo".to_owned()), None::<bool>);
}
