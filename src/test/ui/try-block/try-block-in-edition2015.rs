// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: --edition 2015

pub fn main() {
    let try_result: Option<_> = try {
    //~^ ERROR expected struct, variant or union type, found macro `try`
        let x = 5; //~ ERROR expected identifier, found keyword
        x
    };
    assert_eq!(try_result, Some(5));
}
