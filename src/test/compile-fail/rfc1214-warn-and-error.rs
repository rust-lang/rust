// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that an RFC1214 warning from an earlier function (`foo`) does
// not suppress an error for the same problem (`WantEq<NotEq>`,
// `NotEq: !Eq`) in a later function (`bar)`. Earlier versions of the
// warning mechanism had an issue due to caching.

#![allow(dead_code)]
#![allow(unused_variables)]

struct WantEq<T:Eq> { t: T }

struct NotEq;

trait Trait<T> { }

fn foo() {
    let x: Box<Trait<WantEq<NotEq>>> = loop { };
    //~^ WARN E0277
}

fn bar() {
    wf::<WantEq<NotEq>>();
    //~^ ERROR E0277
}

fn wf<T>() { }

fn main() { }
