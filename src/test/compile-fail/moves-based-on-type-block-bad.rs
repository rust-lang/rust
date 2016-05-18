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

#![feature(box_patterns)]
#![feature(box_syntax)]

struct S {
    x: Box<E>
}

enum E {
    Foo(Box<S>),
    Bar(Box<isize>),
    Baz
}

fn f<G>(s: &S, g: G) where G: FnOnce(&S) {
    g(s)
}

fn main() {
    let s = S { x: box E::Bar(box 42) };
    loop {
        f(&s, |hellothere| {
            match hellothere.x { //~ ERROR cannot move out
                                 //~| cannot move out of borrowed content
                box E::Foo(_) => {}
                box E::Bar(x) => println!("{}", x.to_string()),
                //~^ NOTE to prevent move
                box E::Baz => {}
            }
        })
    }
}
