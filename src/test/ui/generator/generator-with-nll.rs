// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -Z borrowck=compare

#![feature(generators)]
#![feature(nll)]

fn main() {
    || {
        // The reference in `_a` is a Legal with NLL since it ends before the yield
        let _a = &mut true; //~ ERROR borrow may still be in use when generator yields (Ast)
        let b = &mut true; //~ ERROR borrow may still be in use when generator yields (Ast)
        //~^ borrow may still be in use when generator yields (Mir)
        yield ();
        println!("{}", b);
    };
}
