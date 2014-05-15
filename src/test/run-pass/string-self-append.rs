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
    // Make sure we properly handle repeated self-appends.
    let mut a: StrBuf = "A".to_strbuf();
    let mut i = 20;
    let mut expected_len = 1u;
    while i > 0 {
        println!("{}", a.len());
        assert_eq!(a.len(), expected_len);
        a = format_strbuf!("{}{}", a, a);
        i -= 1;
        expected_len *= 2u;
    }
}
