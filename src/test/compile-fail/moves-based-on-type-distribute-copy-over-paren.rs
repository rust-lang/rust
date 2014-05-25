// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests that references to move-by-default values trigger moves when
// they occur as part of various kinds of expressions.

struct Foo<A> { f: A }
fn touch<A>(_a: &A) {}

fn f00() {
    let x = "hi".to_string();
    let _y = Foo { f:x }; //~ NOTE `x` moved here
    touch(&x); //~ ERROR use of moved value: `x`
}

fn f05() {
    let x = "hi".to_string();
    let _y = Foo { f:(((x))) }; //~ NOTE `x` moved here
    touch(&x); //~ ERROR use of moved value: `x`
}

fn f10() {
    let x = "hi".to_string();
    let _y = Foo { f:x.clone() };
    touch(&x);
}

fn f20() {
    let x = "hi".to_string();
    let _y = Foo { f:(x).clone() };
    touch(&x);
}

fn f30() {
    let x = "hi".to_string();
    let _y = Foo { f:((x)).clone() };
    touch(&x);
}

fn f40() {
    let x = "hi".to_string();
    let _y = Foo { f:(((((((x)).clone()))))) };
    touch(&x);
}

fn main() {}
