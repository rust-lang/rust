// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we correctly infer that `E` must be `()` here.  This is
// known because there is just one impl that could apply where
// `Self=()`.

pub trait FromError<E> {
    fn from_error(err: E) -> Self;
}

impl<E> FromError<E> for E {
    fn from_error(err: E) -> E {
        err
    }
}

fn test() -> Result<(), ()> {
    Err(FromError::from_error(()))
}

fn main() {
    let result = (|| Err(FromError::from_error(())))();
    let foo: () = result.unwrap_or(());
}
