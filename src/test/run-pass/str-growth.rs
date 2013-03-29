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
    let mut s = ~"a";
    s += ~"b";
    assert!((s[0] == 'a' as u8));
    assert!((s[1] == 'b' as u8));
    s += ~"c";
    s += ~"d";
    assert!((s[0] == 'a' as u8));
    assert!((s[1] == 'b' as u8));
    assert!((s[2] == 'c' as u8));
    assert!((s[3] == 'd' as u8));
}
