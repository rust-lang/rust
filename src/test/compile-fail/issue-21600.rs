// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn call_it<F>(f: F) where F: Fn() { f(); }

struct A;

impl A {
    fn gen(&self) {}
    fn gen_mut(&mut self) {}
}

fn main() {
    let mut x = A;
    call_it(|| {    //~ HELP consider changing this to accept closures that implement `FnMut`
        call_it(|| x.gen());
        call_it(|| x.gen_mut()); //~ ERROR cannot borrow data mutably in a captured outer
        //~^ ERROR cannot borrow data mutably in a captured outer
        //~| HELP consider changing this closure to take self by mutable reference
    });
}
