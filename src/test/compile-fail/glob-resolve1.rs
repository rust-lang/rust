// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Make sure that globs only bring in public things.

#![feature(globs)]

use bar::*;

mod bar {
    use import = self::fpriv;
    fn fpriv() {}
    extern {
        fn epriv();
    }
    enum A { A1 }
    pub enum B { B1 }

    struct C;

    type D = int;
}

fn foo<T>() {}

fn main() {
    fpriv(); //~ ERROR: unresolved
    epriv(); //~ ERROR: unresolved
    A1; //~ ERROR: unresolved
    B1;
    C; //~ ERROR: unresolved
    import(); //~ ERROR: unresolved

    foo::<A>(); //~ ERROR: undeclared
    //~^ ERROR: undeclared
    foo::<C>(); //~ ERROR: undeclared
    //~^ ERROR: undeclared
    foo::<D>(); //~ ERROR: undeclared
    //~^ ERROR: undeclared
}
