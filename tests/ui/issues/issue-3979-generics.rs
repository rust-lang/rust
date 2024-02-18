//@ run-pass
#![allow(dead_code)]
#![allow(non_snake_case)]

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

struct Point { x: isize, y: isize }

impl Positioned<isize> for Point {
    fn SetX(&mut self, x: isize) {
        self.x = x;
    }
    fn X(&self) -> isize {
        self.x
    }
}

impl Movable<isize> for Point {}

pub fn main() {
    let mut p = Point{ x: 1, y: 2};
    p.translate(3);
    assert_eq!(p.X(), 4);
}
