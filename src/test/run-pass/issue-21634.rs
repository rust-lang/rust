// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


// pretty-expanded FIXME #23616

fn main() {
    if let Ok(x) = "3.1415".parse() {
        assert_eq!(false, x <= 0.0);
    }
    if let Ok(x) = "3.1415".parse() {
        assert_eq!(3.1415, x + 0.0);
    }
    if let Ok(mut x) = "3.1415".parse() {
        assert_eq!(8.1415, { x += 5.0; x });
    }
}
