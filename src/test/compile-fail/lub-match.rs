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
    match *maybestr {
        Some(ref s) => {
            let s: &'a str = s.as_slice();
            s
        }
        None => "(none)",
    }
}

pub fn opt_str1<'a>(maybestr: &'a Option<String>) -> &'a str {
    match *maybestr {
        None => "(none)",
        Some(ref s) => {
            let s: &'a str = s.as_slice();
            s
        }
    }
}

pub fn opt_str2<'a>(maybestr: &'a Option<String>) -> &'static str {
    match *maybestr { //~ ERROR mismatched types
        None => "(none)",
        Some(ref s) => {
            let s: &'a str = s.as_slice();
            s
        }
    }
}

pub fn opt_str3<'a>(maybestr: &'a Option<String>) -> &'static str {
    match *maybestr { //~ ERROR mismatched types
        Some(ref s) => {
            let s: &'a str = s.as_slice();
            s
        }
        None => "(none)",
    }
}

fn main() {}
