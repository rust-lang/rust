// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct r {
  i: @mut int,
}

#[unsafe_destructor]
impl Drop for r {
    fn drop(&self) {
        unsafe {
            *(self.i) = *(self.i) + 1;
        }
    }
}

fn f<T>(_i: ~[T], _j: ~[T]) {
}

fn main() {
    let i1 = @mut 0;
    let i2 = @mut 1;
    let r1 = ~[~r { i: i1 }];
    let r2 = ~[~r { i: i2 }];
    f(r1.clone(), r2.clone());
    //~^ ERROR failed to find an implementation of
    info!((r2, *i1));
    info!((r1, *i2));
}
