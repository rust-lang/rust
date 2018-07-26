// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![deny(unused_parens)]

#[derive(Eq, PartialEq)]
struct X { y: bool }
impl X {
    fn foo(&self, conjunct: bool) -> bool { self.y && conjunct }
}

fn foo() -> isize {
    return (1); //~ ERROR unnecessary parentheses around `return` value
}
fn bar(y: bool) -> X {
    return (X { y }); //~ ERROR unnecessary parentheses around `return` value
}

fn main() {
    foo();
    bar((true)); //~ ERROR unnecessary parentheses around function argument

    if (true) {} //~ ERROR unnecessary parentheses around `if` condition
    while (true) {} //~ ERROR unnecessary parentheses around `while` condition
    match (true) { //~ ERROR unnecessary parentheses around `match` head expression
        _ => {}
    }
    if let 1 = (1) {} //~ ERROR unnecessary parentheses around `if let` head expression
    while let 1 = (2) {} //~ ERROR unnecessary parentheses around `while let` head expression
    let v = X { y: false };
    // struct lits needs parens, so these shouldn't warn.
    if (v == X { y: true }) {}
    if (X { y: true } == v) {}
    if (X { y: false }.y) {}

    while (X { y: false }.foo(true)) {}
    while (true | X { y: false }.y) {}

    match (X { y: false }) {
        _ => {}
    }

    X { y: false }.foo((true)); //~ ERROR unnecessary parentheses around method argument

    let mut _a = (0); //~ ERROR unnecessary parentheses around assigned value
    _a = (0); //~ ERROR unnecessary parentheses around assigned value
    _a += (1); //~ ERROR unnecessary parentheses around assigned value
}
