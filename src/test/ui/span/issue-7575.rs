// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test the mechanism for warning about possible missing `self` declarations.
// ignore-tidy-linelength

trait CtxtFn {
    fn f8(self, usize) -> usize;
    fn f9(usize) -> usize; //~ NOTE candidate
}

trait OtherTrait {
    fn f9(usize) -> usize; //~ NOTE candidate
}

// Note: this trait is not implemented, but we can't really tell
// whether or not an impl would match anyhow without a self
// declaration to match against, so we wind up prisizeing it as a
// candidate. This seems not unreasonable -- perhaps the user meant to
// implement it, after all.
trait UnusedTrait {
    fn f9(usize) -> usize; //~ NOTE candidate
}

impl CtxtFn for usize {
    fn f8(self, i: usize) -> usize {
        i * 4
    }

    fn f9(i: usize) -> usize {
        i * 4
    }
}

impl OtherTrait for usize {
    fn f9(i: usize) -> usize {
        i * 8
    }
}

struct Myisize(isize);

impl Myisize {
    fn fff(i: isize) -> isize { //~ NOTE candidate
        i
    }
}

trait ManyImplTrait {
    fn is_str() -> bool { //~ NOTE candidate
        false
    }
}

impl ManyImplTrait for String {
    fn is_str() -> bool {
        true
    }
}

impl ManyImplTrait for usize {}
impl ManyImplTrait for isize {}
impl ManyImplTrait for char {}
impl ManyImplTrait for Myisize {}

fn no_param_bound(u: usize, m: Myisize) -> usize {
    u.f8(42) + u.f9(342) + m.fff(42)
            //~^ ERROR no method named `f9` found for type `usize` in the current scope
            //~| NOTE found the following associated functions; to be used as methods, functions must have a `self` parameter
            //~| NOTE to use it here write `CtxtFn::f9(u, 342)` instead
            //~| ERROR no method named `fff` found for type `Myisize` in the current scope
            //~| NOTE found the following associated functions; to be used as methods, functions must have a `self` parameter
            //~| NOTE to use it here write `OtherTrait::f9(u, 342)` instead
            //~| NOTE to use it here write `UnusedTrait::f9(u, 342)` instead
}

fn param_bound<T: ManyImplTrait>(t: T) -> bool {
    t.is_str()
    //~^ ERROR no method named `is_str` found for type `T` in the current scope
    //~| NOTE found the following associated functions; to be used as methods, functions must have a `self` parameter
    //~| NOTE to use it here write `ManyImplTrait::is_str(t)` instead
}

fn main() {
}
