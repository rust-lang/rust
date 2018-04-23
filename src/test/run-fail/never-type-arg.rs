// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we can use ! as an argument to a trait impl.

// error-pattern:oh no!

#![feature(never_type)]

struct Wub;

impl PartialEq<!> for Wub {
    fn eq(&self, other: &!) -> bool {
        *other
    }
}

fn main() {
    let _ = Wub == panic!("oh no!");
}

