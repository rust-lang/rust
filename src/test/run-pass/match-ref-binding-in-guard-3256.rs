// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::unstable;

pub fn main() {
    unsafe {
        let x = Some(unstable::sync::Exclusive::new(true));
        match x {
            Some(ref z) if z.with(|b| *b) => {
                do z.with |b| { assert!(*b); }
            },
            _ => fail!()
        }
    }
}
