// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:cci_class_4.rs
extern crate cci_class_4;
use cci_class_4::kitties::cat;

pub fn main() {
    let mut nyan = cat(0u, 2, "nyan".to_strbuf());
    nyan.eat();
    assert!((!nyan.eat()));
    for _ in range(1u, 10u) { nyan.speak(); };
    assert!((nyan.eat()));
}
