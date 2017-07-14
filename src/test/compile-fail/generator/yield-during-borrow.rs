// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(generators, generator_trait)]

use std::ops::{State, Generator};
use std::cell::Cell;

fn borrow_local_inline() {
    // Not OK to yield with a borrow of a temporary.
    //
    // (This error occurs because the region shows up in the type of
    // `b` and gets extended by region inference.)
    let mut b = move || {
        let a = &3;
        yield();
        println!("{}", a);
    }; //~ ERROR E0597
    b.resume();
}

fn borrow_local_inline_done() {
    // No error here -- `a` is not in scope at the point of `yield`.
    let mut b = move || {
        {
            let a = &3;
        }
        yield();
    };
    b.resume();
}

fn borrow_local() {
    // Not OK to yield with a borrow of a temporary.
    //
    // (This error occurs because the region shows up in the type of
    // `b` and gets extended by region inference.)
    let mut b = move || {
        let a = 3;
        {
            let b = &a;
            yield();
            println!("{}", b);
        }
    }; //~ ERROR E0597
    b.resume();
}

fn reborrow_shared_ref(x: &i32) {
    // This is OK -- we have a borrow live over the yield, but it's of
    // data that outlives the generator.
    let mut b = move || {
        let a = &*x;
        yield();
        println!("{}", a);
    };
    b.resume();
}

fn reborrow_mutable_ref(x: &mut i32) {
    // This is OK -- we have a borrow live over the yield, but it's of
    // data that outlives the generator.
    let mut b = move || {
        let a = &mut *x;
        yield();
        println!("{}", a);
    };
    b.resume();
}

fn reborrow_mutable_ref_2(x: &mut i32) {
    // ...but not OK to go on using `x`.
    let mut b = || {
        let a = &mut *x;
        yield();
        println!("{}", a);
    };
    println!("{}", x); //~ ERROR E0501
    b.resume();
}

fn yield_during_iter_owned_data(x: Vec<i32>) {
    // The generator owns `x`, so we error out when yielding with a
    // reference to it.  This winds up becoming a rather confusing
    // regionck error -- in particular, we would freeze with the
    // reference in scope, and it doesn't live long enough.
    let _b = move || {
        for p in &x {
            yield();
        }
    }; //~ ERROR E0597
}

fn yield_during_iter_borrowed_slice(x: &[i32]) {
    let _b = move || {
        for p in x {
            yield();
        }
    };
}

fn yield_during_iter_borrowed_slice_2() {
    let mut x = vec![22_i32];
    let _b = || {
        for p in &x {
            yield();
        }
    };
    println!("{:?}", x);
}

fn yield_during_iter_borrowed_slice_3() {
    // OK to take a mutable ref to `x` and yield
    // up pointers from it:
    let mut x = vec![22_i32];
    let mut b = || {
        for p in &mut x {
            yield p;
        }
    };
    b.resume();
}

fn yield_during_iter_borrowed_slice_4() {
    // ...but not OK to do that while reading
    // from `x` too
    let mut x = vec![22_i32];
    let mut b = || {
        for p in &mut x {
            yield p;
        }
    };
    println!("{}", x[0]); //~ ERROR cannot borrow `x` as immutable
    b.resume();
}

fn main() { }
