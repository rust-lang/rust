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

trait CtxtFn {
    fn f8(self, uint) -> uint;
    fn f9(uint) -> uint; //~ NOTE candidate
}

trait OtherTrait {
    fn f9(uint) -> uint; //~ NOTE candidate
}

// Note: this trait is not implemented, but we can't really tell
// whether or not an impl would match anyhow without a self
// declaration to match against, so we wind up printing it as a
// candidate. This seems not unreasonable -- perhaps the user meant to
// implement it, after all.
trait UnusedTrait {
    fn f9(uint) -> uint; //~ NOTE candidate
}

impl CtxtFn for uint {
    fn f8(self, i: uint) -> uint {
        i * 4u
    }

    fn f9(i: uint) -> uint {
        i * 4u
    }
}

impl OtherTrait for uint {
    fn f9(i: uint) -> uint {
        i * 8u
    }
}

struct MyInt(int);

impl MyInt {
    fn fff(i: int) -> int { //~ NOTE candidate
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

impl ManyImplTrait for uint {}
impl ManyImplTrait for int {}
impl ManyImplTrait for char {}
impl ManyImplTrait for MyInt {}

fn no_param_bound(u: uint, m: MyInt) -> uint {
    u.f8(42) + u.f9(342) + m.fff(42)
            //~^ ERROR type `uint` does not implement any method in scope named `f9`
            //~^^ NOTE found defined static methods, maybe a `self` is missing?
                        //~^^^ ERROR type `MyInt` does not implement any method in scope named `fff`
                        //~^^^^ NOTE found defined static methods, maybe a `self` is missing?
}

fn param_bound<T: ManyImplTrait>(t: T) -> bool {
    t.is_str()
    //~^ ERROR type `T` does not implement any method in scope named `is_str`
    //~^^ NOTE found defined static methods, maybe a `self` is missing?
}

fn main() {
}
