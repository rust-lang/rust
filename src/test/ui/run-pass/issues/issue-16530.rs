// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::hash::{SipHasher, Hasher, Hash};

#[derive(Hash)]
struct Empty;

pub fn main() {
    let mut s1 = SipHasher::new_with_keys(0, 0);
    Empty.hash(&mut s1);
    let mut s2 = SipHasher::new_with_keys(0, 0);
    Empty.hash(&mut s2);
    assert_eq!(s1.finish(), s2.finish());
}
