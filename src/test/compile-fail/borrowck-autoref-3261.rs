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
    pub fn with<F>(&self, blk: F) where F: FnOnce(&Either<(uint, uint), fn()>) {
        let X(ref e) = *self;
        blk(e)
    }
}

fn main() {
    let mut x = X(Either::Right(main as fn()));
    (&mut x).with(
        |opt| { //~ ERROR cannot borrow `x` as mutable more than once at a time
            match opt {
                &Either::Right(ref f) => {
                    x = X(Either::Left((0, 0)));
                    (*f)()
                },
                _ => panic!()
            }
        })
}
