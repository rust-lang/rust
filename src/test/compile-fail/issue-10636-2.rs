// Copyright 2013-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// FIXME(31528) we emit a bunch of silly errors here due to continuing past the
// first one. This would be easy-ish to address by better recovery in tokenisation.

pub fn trace_option(option: Option<isize>) {
    option.map(|some| 42; //~ NOTE: unclosed delimiter
                          //~^ ERROR: expected one of
} //~ ERROR: incorrect close delimiter
//~^ ERROR: expected expression, found `)`
