// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
 * This is a regression test for a bug in LLVM, fixed in upstream r179587,
 * where the switch instructions generated for destructuring enums
 * represented with nullable pointers could be misoptimized in some cases.
 */

enum List<X> { Nil, Cons(X, @List<X>) }
pub fn main() {
    match Cons(10, @Nil) {
        Cons(10, _) => {}
        Nil => {}
        _ => fail!()
    }
}
