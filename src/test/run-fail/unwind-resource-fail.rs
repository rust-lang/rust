// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// error-pattern:squirrel

struct r {
    i: int,
}

impl Drop for r {
    fn drop(&self) { fail!("squirrel") }
}

fn r(i: int) -> r { r { i: i } }

fn main() {
    @0;
    let r = r(0);
}
