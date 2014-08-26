// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we correctly consider the type of `match` to be the LUB
// of the various arms, particularly in the case where regions are
// involved.

pub fn opt_str0<'a>(maybestr: &'a Option<String>) -> &'a str {
    if maybestr.is_none() {
        "(none)"
    } else {
        let s: &'a str = maybestr.get_ref().as_slice();
        s
    }
}

pub fn opt_str1<'a>(maybestr: &'a Option<String>) -> &'a str {
    if maybestr.is_some() {
        let s: &'a str = maybestr.get_ref().as_slice();
        s
    } else {
        "(none)"
    }
}

pub fn opt_str2<'a>(maybestr: &'a Option<String>) -> &'static str {
    if maybestr.is_none() {
        "(none)"
    } else {
        let s: &'a str = maybestr.get_ref().as_slice();
        s  //~ ERROR cannot infer an appropriate lifetime for automatic coercion due to conflicting
    }
}

pub fn opt_str3<'a>(maybestr: &'a Option<String>) -> &'static str {
    if maybestr.is_some() {
        let s: &'a str = maybestr.get_ref().as_slice();
        s  //~ ERROR cannot infer an appropriate lifetime for automatic coercion due to conflicting
    } else {
        "(none)"
    }
}


fn main() {}
