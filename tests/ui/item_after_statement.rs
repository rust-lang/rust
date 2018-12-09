// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![warn(clippy::items_after_statements)]

fn ok() {
    fn foo() {
        println!("foo");
    }
    foo();
}

fn last() {
    foo();
    fn foo() {
        println!("foo");
    }
}

fn main() {
    foo();
    fn foo() {
        println!("foo");
    }
    foo();
}

fn mac() {
    let mut a = 5;
    println!("{}", a);
    // do not lint this, because it needs to be after `a`
    macro_rules! b {
        () => {{
            a = 6
        }};
    }
    b!();
    println!("{}", a);
}
