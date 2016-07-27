// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Note: This test is checking that we forbid a coding pattern that
// Issue #5873 explicitly wants to allow.

enum State { ST_NULL, ST_WHITESPACE }

fn main() {
    [State::ST_NULL; (State::ST_WHITESPACE as usize)];
    //~^ ERROR constant evaluation error
    //~| unimplemented constant expression: enum variants
}
