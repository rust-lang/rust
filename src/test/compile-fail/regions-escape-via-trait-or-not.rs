// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait deref {
    fn get(self) -> int;
}

impl<'a> deref for &'a int {
    fn get(self) -> int {
        *self
    }
}

fn with<R:deref>(f: |x: &int| -> R) -> int {
    f(&3).get()
}

fn return_it() -> int {
    with(|o| o) //~ ERROR cannot infer an appropriate lifetime
}

fn main() {
}
