// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn parse_args() -> ~str {
    let args = ::std::os::args();
    let mut n = 0;

    while n < args.len() {
        match args[n].clone() {
            ~"-v" => (),
            s => {
                return s;
            }
        }
        n += 1;
    }

    return ~""
}

pub fn main() {
    println(parse_args());
}
