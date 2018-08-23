// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(existential_type)]
// compile-pass
mod my_mod {
  use std::fmt::Debug;

  pub existential type Foo: Debug;
  pub existential type Foot: Debug;

  pub fn get_foo() -> Foo {
      5i32
  }

  pub fn get_foot() -> Foot {
      get_foo()
  }
}

fn main() {
    let _: my_mod::Foot = my_mod::get_foot();
}

