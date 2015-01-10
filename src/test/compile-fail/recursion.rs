// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-command-line: See https://github.com/rust-lang/rust/issues/20747
//
// We also get a second error message at the top of file (dummy
// span). This is not helpful, but also kind of annoying to prevent,
// so for now just live with it, since we also get a second message
// that is more helpful.

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
  match n {    0 => {first.dot(second)}
      //~^ ERROR: reached the recursion limit during monomorphization
      // Error message should be here. It should be a type error
      // to instantiate `test` at a type other than T. (See #4287)
    _ => {test (n-1, i+1, Cons {head:2*i+1, tail:first}, Cons{head:i*i, tail:second})}
  }
}
pub fn main() {
  let n = test(1, 0, Nil::NilValue, Nil::NilValue);
  println!("{}", n);
}
