// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn read_lines_borrowed() -> Vec<&str> {
    let raw_lines: Vec<String> = vec!("foo  ".to_string(), "  bar".to_string());
    raw_lines.iter().map(|l| l.as_slice().trim()).collect()
    //~^ ERROR `raw_lines` does not live long enough
}

fn main() {
    println!("{}", read_lines_borrowed());
}
