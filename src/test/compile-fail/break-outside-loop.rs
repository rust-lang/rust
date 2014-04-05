// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Foo {
    t: ~str
}

fn cond() -> bool { true }

fn foo(_: ||) {}

fn main() {
    let pth = break; //~ ERROR: `break` outside of loop
    if cond() { continue } //~ ERROR: `continue` outside of loop

    while cond() {
        if cond() { break }
        if cond() { continue }
        foo(|| {
            if cond() { break } //~ ERROR: `break` inside of a closure
            if cond() { continue } //~ ERROR: `continue` inside of a closure
        })
    }

    let rs: Foo = Foo{t: pth};

    let unconstrained = break; //~ ERROR: `break` outside of loop
}
