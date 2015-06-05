// Copyright 2014-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(macro_rules)]

macro_rules! g {
    ($inp:ident) => (
        { $inp $nonexistent }
        //~^ ERROR unknown macro variable `nonexistent`
        //~| ERROR macro expansion ignores token `$nonexistent` and any following
    );
}

fn main() {
    g!(foo);
//~^ NOTE caused by the macro expansion here; the usage of `g` is likely invalid in this context
}
