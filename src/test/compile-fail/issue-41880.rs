// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn iterate<T, F>(initial: T, f: F) -> Iterate<T, F> {
    Iterate {
        state: initial,
        f: f,
    }
}

pub struct Iterate<T, F> {
    state: T,
    f: F
}

impl<T: Clone, F> Iterator for Iterate<T, F> where F: Fn(&T) -> T {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.state = (self.f)(&self.state);
        Some(self.state.clone())
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) { (std::usize::MAX, None) }
}

fn main() {
    let a = iterate(0, |x| x+1);
    println!("{:?}", a.iter().take(10).collect::<Vec<usize>>());
    //~^ ERROR no method named `iter` found for type `Iterate<{integer}
}
