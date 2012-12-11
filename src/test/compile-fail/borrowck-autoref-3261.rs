// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use either::*;
enum X = Either<(uint,uint),fn()>;
impl &X {
    fn with(blk: fn(x: &Either<(uint,uint),fn()>)) {
        blk(&**self)
    }
}
fn main() {
    let mut x = X(Right(main));
    do (&mut x).with |opt| {  //~ ERROR illegal borrow
        match *opt {
            Right(f) => {
                x = X(Left((0,0)));
                f()
            },
            _ => fail
        }
    }
}