// Regression test for https://github.com/rust-lang/rust/issues/166466

#![feature(gca_min_const_items, function_arg_const_generics)]

struct C {}

impl C {
    pub fn speak<T>(&mut self, const N: usize ) {}
}

pub fn main() {
  let mut nyan = C {};
  nyan.speak(vec![1,2,3]);
  //~^ ERROR
  //~| ERROR E0308
}
