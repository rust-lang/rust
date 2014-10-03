// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern crate debug;

struct MyStruct;

trait Repro {
  fn repro(self, s: MyStruct) -> String;
}

impl Repro for |MyStruct|:'static -> String {
  fn repro(self, s: MyStruct) -> String {
    self(s)
  }
}

fn do_stuff<R: Repro>(r: R) -> String {
  r.repro(MyStruct)
}

pub fn main() {
  assert_eq!("MyStruct".to_string(), do_stuff(|s: MyStruct| format!("{:?}", s)));
}
