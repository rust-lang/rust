// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct dtor {
    x: @mut int,

}

impl dtor : Drop {
    fn finalize(&self) {
        // abuse access to shared mutable state to write this code
        *self.x -= 1;
    }
}

fn unwrap<T>(+o: Option<T>) -> T {
    match move o {
      Some(move v) => move v,
      None => fail
    }
}

fn main() {
    let x = @mut 1;

    {
        let b = Some(dtor { x:x });
        let c = unwrap(move b);
    }

    assert *x == 0;
}
