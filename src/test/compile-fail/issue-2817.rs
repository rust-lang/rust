// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::uint;

fn uuid() -> uint { fail!(); }

fn from_str(s: ~str) -> uint { fail!(); }
fn to_str(u: uint) -> ~str { fail!(); }
fn uuid_random() -> uint { fail!(); }

fn main() {
    do range(0u, 100000).advance |_i| { //~ ERROR Do-block body must return bool, but
    };
    // should get a more general message if the callback
    // doesn't return nil
    do range(0u, 100000).advance |_i| { //~ ERROR mismatched types
        ~"str"
    };
}
