// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

enum Either<T, U> { Left(T), Right(U) }

struct X(Either<(uint,uint), fn()>);

impl X {
    pub fn with(&self, blk: |x: &Either<(uint,uint), fn()>|) {
        let X(ref e) = *self;
        blk(e)
    }
}

fn main() {
    let mut x = X(Right(main));
    let x_ptr = &mut x;
    x_ptr.with(
        |opt| { //~ ERROR closure requires unique access
            match opt {
                &Right(ref f) => {
                    *x_ptr = X(Left((0,0)));    //~ ERROR cannot move `x_ptr`
                    (*f)()
                },
                _ => fail!()
            }
        })
}
