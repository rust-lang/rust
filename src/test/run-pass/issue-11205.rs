// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// pretty-expanded FIXME #23616

#![allow(dead_code)]

// FIXME (#22405): Replace `Box::new` with `box` here when/if possible.

trait Foo { fn dummy(&self) { } }
impl Foo for isize {}
fn foo(_: [&Foo; 2]) {}
fn foos(_: &[&Foo]) {}
fn foog<T>(_: &[T], _: &[T]) {}

fn bar(_: [Box<Foo>; 2]) {}
fn bars(_: &[Box<Foo>]) {}

fn main() {
    let x: [&Foo; 2] = [&1, &2];
    foo(x);
    foo([&1, &2]);

    let r = &1;
    let x: [&Foo; 2] = [r; 2];
    foo(x);
    foo([&1; 2]);

    let x: &[&Foo] = &[&1, &2];
    foos(x);
    foos(&[&1, &2]);

    let x: &[&Foo] = &[&1, &2];
    let r = &1;
    foog(x, &[r]);

    let x: [Box<Foo>; 2] = [Box::new(1), Box::new(2)];
    bar(x);
    bar([Box::new(1), Box::new(2)]);

    let x: &[Box<Foo>] = &[Box::new(1), Box::new(2)];
    bars(x);
    bars(&[Box::new(1), Box::new(2)]);

    let x: &[Box<Foo>] = &[Box::new(1), Box::new(2)];
    foog(x, &[Box::new(1)]);

    struct T<'a> {
        t: [&'a (Foo+'a); 2]
    }
    let _n = T {
        t: [&1, &2]
    };
    let r = &1;
    let _n = T {
        t: [r; 2]
    };
    let x: [&Foo; 2] = [&1, &2];
    let _n = T {
        t: x
    };

    struct F<'b> {
        t: &'b [&'b (Foo+'b)]
    }
    let _n = F {
        t: &[&1, &2]
    };
    let r = &1;
    let r: [&Foo; 2] = [r; 2];
    let _n = F {
        t: &r
    };
    let x: [&Foo; 2] = [&1, &2];
    let _n = F {
        t: &x
    };

    struct M<'a> {
        t: &'a [Box<Foo+'static>]
    }
    let _n = M {
        t: &[Box::new(1), Box::new(2)]
    };
    let x: [Box<Foo>; 2] = [Box::new(1), Box::new(2)];
    let _n = M {
        t: &x
    };
}
