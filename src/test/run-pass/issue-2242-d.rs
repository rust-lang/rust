// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-fast (aux-build)
// aux-build:issue_2242_a.rs
// aux-build:issue_2242_b.rs
// aux-build:issue_2242_c.rs

extern mod a;
extern mod b;
extern mod c;

use a::to_strz;

fn main() {
    io::println((~"foo").to_strz());
    io::println(1.to_strz());
    io::println(true.to_strz());
}
