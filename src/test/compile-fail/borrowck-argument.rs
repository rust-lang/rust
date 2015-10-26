// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[derive(Copy, Clone)]
struct S;

impl S {
    fn mutate(&mut self) {
    }
}

fn func(arg: S) {
    arg.mutate(); //~ ERROR: cannot borrow immutable argument
}

impl S {
    fn method(&self, arg: S) {
        arg.mutate(); //~ ERROR: cannot borrow immutable argument
    }
}

trait T {
    fn default(&self, arg: S) {
        arg.mutate(); //~ ERROR: cannot borrow immutable argument
    }
}

impl T for S {}

fn main() {
    let s = S;
    func(s);
    s.method(s);
    s.default(s);
    (|arg: S| { arg.mutate() })(s); //~ ERROR: cannot borrow immutable argument
}
