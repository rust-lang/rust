// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// error-pattern: copying a noncopyable value

struct r {
  i: @mut int,
}

impl r : Drop {
    fn finalize(&self) {
        *(self.i) = *(self.i) + 1;
    }
}

fn f<T>(+i: ~[T], +j: ~[T]) {
}

fn main() {
    let i1 = @mut 0;
    let i2 = @mut 1;
    let r1 = move ~[~r { i: i1 }];
    let r2 = move ~[~r { i: i2 }];
    f(copy r1, copy r2);
    log(debug, (r2, *i1));
    log(debug, (r1, *i2));
}
