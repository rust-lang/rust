// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    let mut c = (1, (1, "".to_owned()));
    match c {
        c2 => { (c.1).0 = 2; assert_eq!((c2.1).0, 1); }
    }

    let mut c = (1, (1, (1, "".to_owned())));
    match c.1 {
        c2 => { ((c.1).1).0 = 3; assert_eq!((c2.1).0, 1); }
    }
}
