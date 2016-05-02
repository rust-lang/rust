// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct NoCopy;
fn main() {
   let x = NoCopy;
   let f = move || { let y = x; };
   //~^ value moved (into closure) here
   let z = x;
   //~^ ERROR use of moved value: `x`
   //~| value used here after move
   //~| move occurs because `x` has type `NoCopy`
}
