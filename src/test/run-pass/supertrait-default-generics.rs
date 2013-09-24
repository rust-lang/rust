// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// There is some other borrowck bug, so we make the stuff not mut.

trait Positioned<S> {
  fn SetX(&mut self, S);
  fn X(&self) -> S;
}

trait Movable<S: Add<S, S>>: Positioned<S> {
  fn translate(&mut self, dx: S) {
    let x = self.X() + dx;
    self.SetX(x);
  }
}

struct Point<S> { x: S, y: S }

impl<S: Clone> Positioned<S> for Point<S> {
    fn SetX(&mut self, x: S) {
        self.x = x;
    }
    fn X(&self) -> S {
        self.x.clone()
    }
}

impl<S: Clone + Add<S, S>> Movable<S> for Point<S> {}

pub fn main() {
    let mut p = Point{ x: 1, y: 2};
    p.translate(3);
    assert_eq!(p.X(), 4);
}
