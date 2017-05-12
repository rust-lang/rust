// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(fn_traits)]

// Ensure that invoking a closure counts as a unique immutable borrow

type Fn<'a> = Box<FnMut() + 'a>;

struct Test<'a> {
    f: Box<FnMut() + 'a>
}

fn call<F>(mut f: F) where F: FnMut(Fn) {
    f(Box::new(|| {
    //~^ ERROR: cannot borrow `f` as mutable more than once
    //~| NOTE first mutable borrow occurs here
    //~| NOTE second mutable borrow occurs here
        f((Box::new(|| {})))
    }));
    //~^ NOTE first borrow ends here
}

fn test1() {
    call(|mut a| {
        a.call_mut(());
    });
}

fn test2<F>(f: &F) where F: FnMut() {
    //~^ NOTE use `&mut F` here to make mutable
    (*f)();
    //~^ ERROR cannot borrow immutable borrowed content `*f` as mutable
    //~| NOTE cannot borrow as mutable
}

fn test3<F>(f: &mut F) where F: FnMut() {
    (*f)();
}

fn test4(f: &Test) {
    //~^ NOTE use `&mut Test` here to make mutable
    f.f.call_mut(())
    //~^ ERROR: cannot borrow immutable `Box` content `*f.f` as mutable
    //~| NOTE cannot borrow as mutable
}

fn test5(f: &mut Test) {
    f.f.call_mut(())
}

fn test6() {
    let mut f = || {};
    (|| {
        f();
    })();
}

fn test7() {
    fn foo<F>(_: F) where F: FnMut(Box<FnMut(isize)>, isize) {}
    let mut f = |g: Box<FnMut(isize)>, b: isize| {};
    //~^ NOTE moved
    f(Box::new(|a| {
    //~^ NOTE borrow of `f` occurs here
        foo(f);
        //~^ ERROR cannot move `f` into closure because it is borrowed
        //~| ERROR cannot move out of captured outer variable in an `FnMut` closure
        //~| NOTE move into closure occurs here
        //~| NOTE cannot move out of captured outer variable in an `FnMut` closure
    }), 3);
}

fn main() {}
