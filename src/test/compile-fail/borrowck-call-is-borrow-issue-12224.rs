// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Ensure that invoking a closure counts as a unique immutable borrow


type Fn<'a> = ||:'a;

struct Test<'a> {
    f: ||: 'a
}

fn call(f: |Fn|) {
    f(|| {
    //~^ ERROR: closure requires unique access to `f` but it is already borrowed
        f(|| {})
    });
}

fn test1() {
    call(|a| {
        a();
    });
}

fn test2(f: &||) {
    (*f)(); //~ ERROR: closure invocation in a `&` reference
}

fn test3(f: &mut ||) {
    (*f)();
}

fn test4(f: &Test) {
    (f.f)() //~ ERROR: closure invocation in a `&` reference
}

fn test5(f: &mut Test) {
    (f.f)()
}

fn test6() {
    let f = || {};
    (|| {
        f();
    })();
}

fn test7() {
    fn foo(_: |g: |int|, b: int|) {}
    let f = |g: |int|, b: int| {};
    f(|a| { //~ ERROR: cannot borrow `f` as immutable because previous closure
        foo(f); //~ ERROR: cannot move out of captured outer variable
    }, 3);
}

fn main() {}
