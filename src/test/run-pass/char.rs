// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.



pub fn main() {
    let c: char = 'x';
    let d: char = 'x';
    fail_unless_eq!(c, 'x');
    fail_unless_eq!('x', c);
    fail_unless_eq!(c, c);
    fail_unless_eq!(c, d);
    fail_unless_eq!(d, c);
    fail_unless_eq!(d, 'x');
    fail_unless_eq!('x', d);
}
