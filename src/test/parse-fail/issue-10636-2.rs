// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
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

// compile-flags: -Z parse-only

pub fn trace_option(option: Option<isize>) { //~ HELP did you mean to close this delimiter?
    option.map(|some| 42; //~ NOTE: unclosed delimiter
                          //~^ ERROR: expected one of
} //~ ERROR: incorrect close delimiter
//~^ ERROR: expected one of
//~ ERROR: this file contains an un-closed delimiter
