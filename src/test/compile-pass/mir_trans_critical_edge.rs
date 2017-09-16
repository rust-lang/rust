// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This code produces a CFG with critical edges that, if we don't
// handle properly, will cause invalid codegen.

#![feature(rustc_attrs)]

enum State {
    Both,
    Front,
    Back
}

pub struct Foo<A: Iterator, B: Iterator> {
    state: State,
    a: A,
    b: B
}

impl<A, B> Foo<A, B>
where A: Iterator, B: Iterator<Item=A::Item>
{
    // This is the function we care about
    fn next(&mut self) -> Option<A::Item> {
        match self.state {
            State::Both => match self.a.next() {
                elt @ Some(..) => elt,
                None => {
                    self.state = State::Back;
                    self.b.next()
                }
            },
            State::Front => self.a.next(),
            State::Back => self.b.next(),
        }
    }
}

// Make sure we actually translate a version of the function
pub fn do_stuff(mut f: Foo<Box<Iterator<Item=u32>>, Box<Iterator<Item=u32>>>) {
    let _x = f.next();
}

fn main() {}
