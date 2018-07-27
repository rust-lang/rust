// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// rust-lang/rust#45696: This test is checking that we can return
// mutable borrows owned by boxes even when the boxes are dropped.
//
// We will explicitly test AST-borrowck, NLL, and migration modes;
// thus we will also skip the automated compare-mode=nll.

// revisions: ast nll migrate
// ignore-compare-mode-nll

#![cfg_attr(nll, feature(nll))]
//[migrate]compile-flags: -Z borrowck=migrate -Z two-phase-borrows

// run-pass

type Boxed<'a, 'b> = Box<(&'a mut u32, &'b mut u32)>;

fn return_borrow_from_dropped_box<'a>(x: Boxed<'a, '_>) -> &'a mut u32 {
    &mut *x.0
}

fn return_borrow_from_dropped_tupled_box<'a>(x: (Boxed<'a, '_>, &mut u32)) -> &'a mut u32 {
    &mut *(x.0).0
}

fn basic_tests() {
    let mut x = 2;
    let mut y = 3;
    let mut z = 4;
    *return_borrow_from_dropped_box(Box::new((&mut x, &mut y))) += 10;
    assert_eq!((x, y, z), (12, 3, 4));
    *return_borrow_from_dropped_tupled_box((Box::new((&mut x, &mut y)), &mut z)) += 10;
    assert_eq!((x, y, z), (22, 3, 4));
}

// These scribbling tests have been transcribed from
// issue-45696-scribble-on-boxed-borrow.rs
//
// In the context of that file, these tests are meant to show cases
// that should be *accepted* by the compiler, so here we are actually
// checking that the code we get when they are compiled matches our
// expectations.

struct Scribble<'a>(&'a mut u32);

impl<'a> Drop for Scribble<'a> { fn drop(&mut self) { *self.0 = 42; } }

// this is okay, in both AST-borrowck and NLL: The `Scribble` here *has*
// to strictly outlive `'a`
fn borrowed_scribble<'a>(s: &'a mut Scribble) -> &'a mut u32 {
    &mut *s.0
}

// this, by analogy to previous case, is also okay.
fn boxed_borrowed_scribble<'a>(s: Box<&'a mut Scribble>) -> &'a mut u32 {
    &mut *(*s).0
}

// this, by analogy to previous case, is also okay.
fn boxed_boxed_borrowed_scribble<'a>(s: Box<Box<&'a mut Scribble>>) -> &'a mut u32 {
    &mut *(**s).0
}

fn scribbling_tests() {
    let mut x = 1;
    {
        let mut long_lived = Scribble(&mut x);
        *borrowed_scribble(&mut long_lived) += 10;
        assert_eq!(*long_lived.0, 11);
        // (Scribble dtor runs here, after `&mut`-borrow above ends)
    }
    assert_eq!(x, 42);
    x = 1;
    {
        let mut long_lived = Scribble(&mut x);
        *boxed_borrowed_scribble(Box::new(&mut long_lived)) += 10;
        assert_eq!(*long_lived.0, 11);
        // (Scribble dtor runs here, after `&mut`-borrow above ends)
    }
    assert_eq!(x, 42);
    x = 1;
    {
        let mut long_lived = Scribble(&mut x);
        *boxed_boxed_borrowed_scribble(Box::new(Box::new(&mut long_lived))) += 10;
        assert_eq!(*long_lived.0, 11);
        // (Scribble dtor runs here, after `&mut`-borrow above ends)
    }
    assert_eq!(x, 42);
}

fn main() {
    basic_tests();
    scribbling_tests();
}
