// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:derive-no-std.rs

extern crate derive_no_std;
use derive_no_std::*;

fn main() {
    let f = Foo { x: 0 };
    assert_eq!(f.clone(), Foo::default());

    assert!(Bar::Qux < Bar::Quux(42));
}

