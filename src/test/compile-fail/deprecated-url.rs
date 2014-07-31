// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-tidy-linelength

#![deny(deprecated)]

extern crate url;

fn main() {
    let _ = url::Url::parse("http://example.com");
    //~^ ERROR use of deprecated item: This is being removed. Use rust-url instead. http://servo.github.io/rust-url/
}
