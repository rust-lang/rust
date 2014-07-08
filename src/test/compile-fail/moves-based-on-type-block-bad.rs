// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-tidy-linelength

struct S {
    x: Box<E>
}

enum E {
    Foo(Box<S>),
    Bar(Box<int>),
    Baz
}

fn f(s: &S, g: |&S|) {
    g(s)
}

fn main() {
    let s = S { x: box Bar(box 42) };
    loop {
        f(&s, |hellothere| {
            match hellothere.x { //~ ERROR cannot move out
                box Foo(_) => {}
                box Bar(x) => println!("{}", x.to_string()), //~ NOTE attempting to move value to here
                box Baz => {}
            }
        })
    }
}
