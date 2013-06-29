// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use std::io;

macro_rules! print_hd_tl (
    ($field_hd:ident, $($field_tl:ident),+) => ({
        io::print(stringify!($field)); //~ ERROR unknown macro variable
        io::print("::[");
        $(
            io::print(stringify!($field_tl));
            io::print(", ");
        )+
        io::print("]\n");
    })
)

fn main() {
    print_hd_tl!(x, y, z, w)
}

