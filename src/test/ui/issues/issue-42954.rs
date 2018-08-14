// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// run-rustfix

#![allow(unused_must_use, unused_comparisons)]

macro_rules! is_plainly_printable {
    ($i: ident) => {
        $i as u32 < 0 //~ `<` is interpreted as a start of generic arguments
    };
}

fn main() {
    let c = 'a';
    is_plainly_printable!(c);
}
