// run-pass
#![allow(dead_code)]
#![allow(non_snake_case)]

// There is some other borrowck bug, so we make the stuff not mut.


use std::ops::Add;

trait Positioned<S> {
  fn SetX(&mut self, _: S);
  fn X(&self) -> S;
}

trait Movable<S: Add<Output=S>>: Positioned<S> {
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

impl<S: Clone + Add<Output=S>> Movable<S> for Point<S> {}

pub fn main() {
    let mut p = Point{ x: 1, y: 2};
    p.translate(3);
    assert_eq!(p.X(), 4);
}
