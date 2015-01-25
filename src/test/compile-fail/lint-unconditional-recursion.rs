// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![deny(unconditional_recursion)]
#![allow(dead_code)]
fn foo() { //~ ERROR function cannot return without recurring
    foo(); //~ NOTE recursive call site
}

fn bar() {
    if true {
        bar()
    }
}

fn baz() { //~ ERROR function cannot return without recurring
    if true {
        baz() //~ NOTE recursive call site
    } else {
        baz() //~ NOTE recursive call site
    }
}

fn qux() {
    loop {}
}

fn quz() -> bool { //~ ERROR function cannot return without recurring
    if true {
        while quz() {} //~ NOTE recursive call site
        true
    } else {
        loop { quz(); } //~ NOTE recursive call site
    }
}

trait Foo {
    fn bar(&self) { //~ ERROR function cannot return without recurring
        self.bar() //~ NOTE recursive call site
    }
}

impl Foo for Box<Foo+'static> {
    fn bar(&self) { //~ ERROR function cannot return without recurring
        loop {
            self.bar() //~ NOTE recursive call site
        }
    }

}

struct Baz;
impl Baz {
    fn qux(&self) { //~ ERROR function cannot return without recurring
        self.qux(); //~ NOTE recursive call site
    }
}

fn main() {}
