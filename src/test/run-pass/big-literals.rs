// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// pretty-expanded FIXME #23616

pub fn main() {
    assert_eq!(0xffffffff, (-1 as u32));
    assert_eq!(4294967295, (-1 as u32));
    assert_eq!(0xffffffffffffffff, (-1 as u64));
    assert_eq!(18446744073709551615, (-1 as u64));

    assert_eq!(-2147483648 - 1, 2147483647);
}
