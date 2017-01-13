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

use bar::*;

mod bar {
    use self::fpriv as import;
    fn fpriv() {}
    extern {
        fn epriv();
    }
    enum A { A1 }
    pub enum B { B1 }

    struct C;

    type D = isize;
}

fn foo<T>() {}

fn main() {
    fpriv(); //~ ERROR cannot find function `fpriv` in this scope
    epriv(); //~ ERROR cannot find function `epriv` in this scope
    B; //~ ERROR expected value, found enum `B`
    C; //~ ERROR cannot find value `C` in this scope
    import(); //~ ERROR: cannot find function `import` in this scope

    foo::<A>(); //~ ERROR: cannot find type `A` in this scope
    foo::<C>(); //~ ERROR: cannot find type `C` in this scope
    foo::<D>(); //~ ERROR: cannot find type `D` in this scope
}
