// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// run-pass
// aux-build:png2.rs
// edition:2018

mod png {
    use png2 as png_ext;

    fn foo() -> png_ext::DecodingError { unimplemented!() }
}

fn main() {
    println!("Hello, world!");
}
