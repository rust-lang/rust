// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

enum Nil {NilValue}
struct Cons<T> {head:isize, tail:T}
trait Dot {fn dot(&self, other:Self) -> isize;}
impl Dot for Nil {
  fn dot(&self, _:Nil) -> isize {0}
}
impl<T:Dot> Dot for Cons<T> {
  fn dot(&self, other:Cons<T>) -> isize {
    self.head * other.head + self.tail.dot(other.tail)
  }
}
fn test<T:Dot> (n:isize, i:isize, first:T, second:T) ->isize {
  match n {    0 => {first.dot(second)} //~ ERROR overflow
      // FIXME(#4287) Error message should be here. It should be
      // a type error to instantiate `test` at a type other than T.
    _ => {test (n-1, i+1, Cons {head:2*i+1, tail:first}, Cons{head:i*i, tail:second})}
  }
}
pub fn main() {
  let n = test(1, 0, Nil::NilValue, Nil::NilValue);
  println!("{}", n);
}
