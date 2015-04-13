#![feature(plugin)]

#![plugin(clippy)]
#![deny(clippy)]
#![allow(unused)]

fn the_answer(ref mut x: u8) {  //~ ERROR `ref` directly on a function argument is ignored
  *x = 42;
}

fn main() {
  let mut x = 0;
  the_answer(x);
  println!("The answer is {}.", x);
}
