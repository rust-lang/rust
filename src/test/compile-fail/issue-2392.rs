// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(core, fnbox)]

use std::boxed::FnBox;

struct FuncContainer {
    f1: fn(data: u8),
    f2: extern "C" fn(data: u8),
    f3: unsafe fn(data: u8),
}

struct FuncContainerOuter {
    container: Box<FuncContainer>
}

struct Obj<F> where F: FnOnce() -> u32 {
    closure: F,
    not_closure: usize,
}

struct BoxedObj {
    boxed_closure: Box<FnBox() -> u32>,
}

struct Wrapper<F> where F: FnMut() -> u32 {
    wrap: Obj<F>,
}

fn func() -> u32 {
    0
}

fn check_expression() -> Obj<Box<FnBox() -> u32>> {
    Obj { closure: Box::new(|| 42_u32) as Box<FnBox() -> u32>, not_closure: 42 }
}

fn main() {
    // test variations of function

    let o_closure = Obj { closure: || 42, not_closure: 42 };
    o_closure.closure(); //~ ERROR no method named `closure` found
    //~^ HELP did you mean to call the function stored in the `closure` field?
    //~| GUESS (o_closure.closure)

    o_closure.not_closure(); //~ ERROR no method named `not_closure` found
    //~| HELP did you mean to write
    //~| GUESS o_closure.not_closure

    let o_func = Obj { closure: func, not_closure: 5 };
    o_func.closure(); //~ ERROR no method named `closure` found
    //~^ HELP did you mean to call the function stored in the `closure` field?
    //~| GUESS (o_func.closure)

    let boxed_fn = BoxedObj { boxed_closure: Box::new(func) };
    boxed_fn.boxed_closure();//~ ERROR no method named `boxed_closure` found
    //~^ HELP did you mean to call the function stored in the `boxed_closure` field?
    //~| GUESS (boxed_fn.boxed_closure)

    let boxed_closure = BoxedObj { boxed_closure: Box::new(|| 42_u32) as Box<FnBox() -> u32> };
    boxed_closure.boxed_closure();//~ ERROR no method named `boxed_closure` found
    //~^ HELP did you mean to call the function stored in the `boxed_closure` field?
    //~| GUESS (boxed_closure.boxed_closure)

    // test expression writing in the notes

    let w = Wrapper { wrap: o_func };
    w.wrap.closure();//~ ERROR no method named `closure` found
    //~| HELP did you mean to call the function stored in
    //~| GUESS (w.wrap.closure)

    w.wrap.not_closure();//~ ERROR no method named `not_closure` found
    //~^ HELP did you mean to write
    //~| w.wrap.not_closure

    check_expression().closure();//~ ERROR no method named `closure` found
    //~| HELP did you mean to call the function stored in
    //~| GUESS (check_expression().closure)
}

impl FuncContainerOuter {
    fn run(&self) {
        unsafe {
            (*self.container).f1(1); //~ ERROR no method named `f1` found
            //~^ HELP did you mean to call the function stored in the `f1` field?
            //~| GUESS ((*self.container).f1)
            (*self.container).f2(1); //~ ERROR no method named `f2` found
            //~^ HELP did you mean to call the function stored in the `f2` field?
            //~| GUESS ((*self.container).f2)
            (*self.container).f3(1); //~ ERROR no method named `f3` found
            //~^ HELP did you mean to call the function stored in the `f3` field?
            //~| GUESS ((*self.container).f3)
        }
    }
}
