// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Foo;
fn takes_ref(_: &Foo) {}

fn main() {
  let ref opt = Some(Foo);
  opt.map(|arg| takes_ref(arg));
  //~^ ERROR mismatched types [E0308]
  opt.and_then(|arg| Some(takes_ref(arg)));
  //~^ ERROR mismatched types [E0308]
  let ref opt: Result<_, ()> = Ok(Foo);
  opt.map(|arg| takes_ref(arg));
  //~^ ERROR mismatched types [E0308]
  opt.and_then(|arg| Ok(takes_ref(arg)));
  //~^ ERROR mismatched types [E0308]
}
