// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(macro_rules)]

macro_rules! print_hd_tl (
    ($field_hd:ident, $($field_tl:ident),+) => ({
        print!("{}", stringify!($field_hd));
        print!("::[");
        $(
            print!("{}", stringify!($field_tl));
            print!(", ");
        )+
        // FIXME: #9970
        print!("{}", "]\n");
    })
)

pub fn main() {
    print_hd_tl!(x, y, z, w)
}

