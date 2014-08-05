// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


fn take_any(_: ||:) {
}

fn take_const_owned(_: ||:Sync+Send) {
}

fn give_any(f: ||:) {
    take_any(f);
}

fn give_owned(f: ||:Send) {
    take_any(f);
    take_const_owned(f); //~ ERROR expected bounds `Send+Sync` but found bounds `Send`
}

fn main() {}
