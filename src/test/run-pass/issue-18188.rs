// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(default_type_params, unboxed_closures)]

use std::thunk::Thunk;

pub trait Promisable: Send + Sync {}
impl<T: Send + Sync> Promisable for T {}
pub fn propagate<T, E, F, G>(action: F) -> Thunk<Result<T, E>, Result<T, E>>
    where
        T: Promisable + Clone,
        E: Promisable + Clone,
        F: FnOnce(&T) -> Result<T, E> + Send,
        G: FnOnce(Result<T, E>) -> Result<T, E> {
    Thunk::with_arg(move |: result: Result<T, E>| {
        match result {
            Ok(ref t) => action(t),
            Err(ref e) => Err(e.clone()),
        }
    })
}

fn main() {}
