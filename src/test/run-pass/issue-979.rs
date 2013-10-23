// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[feature(managed_boxes)];

struct r {
  b: @mut int,
}

#[unsafe_destructor]
impl Drop for r {
    fn drop(&mut self) {
        *(self.b) += 1;
    }
}

fn r(b: @mut int) -> r {
    r {
        b: b
    }
}

pub fn main() {
    let b = @mut 0;
    {
        let _p = Some(r(b));
    }

    assert_eq!(*b, 1);
}
