// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(overloaded_calls)]

use std::ops::FnMut;

fn call_f<F:FnMut<(),()>>(mut f: F) {
    f();
}

fn f() {
    println!("hello");
}

fn call_g<G:FnMut<(String,String),String>>(mut g: G, x: String, y: String)
          -> String {
    g(x, y)
}

fn g(mut x: String, y: String) -> String {
    x.push_str(y.as_slice());
    x
}

fn main() {
    call_f(f);
    assert_eq!(call_g(g, "foo".to_string(), "bar".to_string()).as_slice(),
               "foobar");
}

