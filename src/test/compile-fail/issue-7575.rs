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

use std::marker::MarkerTrait;

trait CtxtFn {
    fn f8(self, usize) -> usize;
    fn f9(usize) -> usize; //~ NOTE candidate
}

trait OtherTrait : MarkerTrait {
    fn f9(usize) -> usize; //~ NOTE candidate
}

// Note: this trait is not implemented, but we can't really tell
// whether or not an impl would match anyhow without a self
// declaration to match against, so we wind up prisizeing it as a
// candidate. This seems not unreasonable -- perhaps the user meant to
// implement it, after all.
trait UnusedTrait : MarkerTrait {
    fn f9(usize) -> usize; //~ NOTE candidate
}

impl CtxtFn for usize {
    fn f8(self, i: usize) -> usize {
        i * 4_usize
    }

    fn f9(i: usize) -> usize {
        i * 4_usize
    }
}

impl OtherTrait for usize {
    fn f9(i: usize) -> usize {
        i * 8_usize
    }
}

struct Myisize(isize);

impl Myisize {
    fn fff(i: isize) -> isize { //~ NOTE candidate
        i
    }
}

trait ManyImplTrait : MarkerTrait {
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
            //~^ ERROR type `usize` does not implement any method in scope named `f9`
            //~^^ NOTE found defined static methods, maybe a `self` is missing?
            //~^^^ ERROR type `Myisize` does not implement any method in scope named `fff`
            //~^^^^ NOTE found defined static methods, maybe a `self` is missing?
}

fn param_bound<T: ManyImplTrait>(t: T) -> bool {
    t.is_str()
    //~^ ERROR type `T` does not implement any method in scope named `is_str`
    //~^^ NOTE found defined static methods, maybe a `self` is missing?
}

fn main() {
}
