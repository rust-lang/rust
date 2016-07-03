// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Future: 'static {
    // The requirement for Self: Sized must prevent instantiation of
    // Future::forget in vtables, otherwise there's an infinite type
    // recursion through <Map<...> as Future>::forget.
    fn forget(self) where Self: Sized {
        Box::new(Map(self)) as Box<Future>;
    }
}

struct Map<A>(A);
impl<A: Future> Future for Map<A> {}

pub struct Promise;
impl Future for Promise {}

fn main() {
    Promise.forget();
}
