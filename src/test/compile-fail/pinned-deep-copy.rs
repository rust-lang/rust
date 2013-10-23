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
  i: @mut int,
}

#[unsafe_destructor]
impl Drop for r {
    fn drop(&mut self) {
        unsafe {
            *(self.i) = *(self.i) + 1;
        }
    }
}

fn r(i: @mut int) -> r {
    r {
        i: i
    }
}

struct A {
    y: r,
}

fn main() {
    let i = @mut 0;
    {
        // Can't do this copy
        let x = ~~~A {y: r(i)};
        let _z = x.clone(); //~ ERROR failed to find an implementation
        info!("{:?}", x);
    }
    error!("{:?}", *i);
}
