//@ run-pass

//@ aux-build:monomorphize-index-op-cross-crate.rs

//! Regression test for https://github.com/rust-lang/rust/issues/2631

extern crate req;

use req::request;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

pub fn main() {
  let v = vec![Rc::new("hi".to_string())];
  let mut m: req::header_map = HashMap::new();
  m.insert("METHOD".to_string(), Rc::new(RefCell::new(v)));
  request::<isize>(&m);
}
