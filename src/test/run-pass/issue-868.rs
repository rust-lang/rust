// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn f<T>(g: &fn() -> T) -> T { g() }

pub fn main() {
  let _x = f( | | { 10 });
    // used to be: cannot determine a type for this expression
    f(| | { });
    // ditto
    f( | | { ()});
    // always worked
    let _: () = f(| | { });
    // empty block with no type info should compile too
    let _ = f(||{});
    let _ = (||{});
}
