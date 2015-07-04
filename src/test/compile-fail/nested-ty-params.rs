// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn hd<U>(v: Vec<U> ) -> U {
    fn hd1(w: [U]) -> U { return w[0]; }
    //~^ ERROR can't use type parameters from outer function; try using a local type
    // parameter instead
    //~| ERROR use of undeclared type name `U`
    //~| ERROR can't use type parameters from outer function; try using a local type
    // parameter instead
    //~| ERROR use of undeclared type name `U`

    return hd1(v);
}
