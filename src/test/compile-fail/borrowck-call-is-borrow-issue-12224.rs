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

#![feature(unboxed_closures)]
#![feature(box_syntax)]

type Fn<'a> = Box<FnMut() + 'a>;

struct Test<'a> {
    f: Box<FnMut() + 'a>
}

fn call<F>(mut f: F) where F: FnMut(Fn) {
    f(box || {
    //~^ ERROR: cannot borrow `f` as mutable more than once
        f(box || {})
    });
}

fn test1() {
    call(|mut a| {
        a.call_mut(());
    });
}

fn test2<F>(f: &F) where F: FnMut() {
    (*f)(); //~ ERROR: cannot borrow immutable borrowed content `*f` as mutable
}

fn test3<F>(f: &mut F) where F: FnMut() {
    (*f)();
}

fn test4(f: &Test) {
    f.f.call_mut(()) //~ ERROR: cannot borrow immutable `Box` content `*f.f` as mutable
}

fn test5(f: &mut Test) {
    f.f.call_mut(())
}

fn test6() {
    let mut f = |&mut:| {};
    (|&mut:| {
        f();
    })();
}

fn test7() {
    fn foo<F>(_: F) where F: FnMut(Box<FnMut(isize)>, isize) {}
    let mut f = |&mut: g: Box<FnMut(isize)>, b: isize| {};
    f(box |a| { //~ ERROR: cannot borrow `f` as immutable because it is also borrowed as mutable
        foo(f); //~ ERROR: cannot move out of captured outer variable
    }, 3);
}

fn main() {}
