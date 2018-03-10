// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// A test for something that NLL enables. It sometimes happens that
// the `while let` pattern makes some borrows from a variable (in this
// case, `x`) that you need in order to compute the next value for
// `x`.  The lexical checker makes this very painful. The NLL checker
// does not.

#![feature(match_default_bindings)]
#![feature(nll)]

use std::rc::Rc;

#[derive(Debug, PartialEq, Eq)]
enum Foo {
    Base(usize),
    Next(Rc<Foo>),
}

fn find_base(mut x: Rc<Foo>) -> Rc<Foo> {
    while let Foo::Next(n) = &*x {
        x = n.clone();
    }
    x
}

fn main() {
    let chain = Rc::new(Foo::Next(Rc::new(Foo::Base(44))));
    let base = find_base(chain);
    assert_eq!(&*base, &Foo::Base(44));
}

