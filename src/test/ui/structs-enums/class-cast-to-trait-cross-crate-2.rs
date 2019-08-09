// run-pass
// aux-build:cci_class_cast.rs

#![feature(box_syntax)]

extern crate cci_class_cast;

use std::string::ToString;
use cci_class_cast::kitty::cat;

fn print_out(thing: Box<dyn ToString>, expected: String) {
  let actual = (*thing).to_string();
  println!("{}", actual);
  assert_eq!(actual.to_string(), expected);
}

pub fn main() {
  let nyan: Box<dyn ToString> = box cat(0, 2, "nyan".to_string()) as Box<dyn ToString>;
  print_out(nyan, "nyan".to_string());
}
