// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Fun {
    type Output;
    fn call<'x>(&'x self) -> Self::Output;
}

struct Holder { x: String }

impl<'a> Fun for Holder {
    type Output = &'a str;
    fn call<'b>(&'b self) -> &'b str {
    //~^ ERROR method `call` has an incompatible type for trait
        &self.x[..]
    }
}

fn main() {}
