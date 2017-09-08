// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that traits can opt-out of being DynSized by default.
// See also run-pass/dynsized/supertrait-default.rs

#![feature(dynsized)]

use std::marker::DynSized;

fn assert_dynsized<T: ?Sized>() { }

trait Tr: ?DynSized {
    fn foo() {
        assert_dynsized::<Self>();
        //~^ ERROR the trait bound `Self: std::marker::DynSized` is not satisfied
    }
}

fn main() { }

