// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[deriving(Show)]
enum chan { chan_t, }

impl Eq for chan {
    fn eq(&self, other: &chan) -> bool {
        ((*self) as uint) == ((*other) as uint)
    }
    fn ne(&self, other: &chan) -> bool { !(*self).eq(other) }
}

fn wrapper3(i: chan) {
    assert_eq!(i, chan_t);
}

pub fn main() {
    let wrapped = {||wrapper3(chan_t)};
    wrapped();
}
