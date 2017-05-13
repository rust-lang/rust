// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that this fairly specialized, but also reasonable, pattern
// typechecks. The pattern involves regions bound in closures that
// wind up related to inference variables.
//
// NB. Changes to the region implementations have broken this pattern
// a few times, but it happens to be used in the compiler so those
// changes were caught. However, those uses in the compiler could
// easily get changed or refactored away in the future.


#![allow(unknown_features)]
#![feature(box_syntax)]

struct Ctxt<'tcx> {
    x: &'tcx Vec<isize>
}

struct Foo<'a,'tcx:'a> {
    cx: &'a Ctxt<'tcx>,
}

impl<'a,'tcx> Foo<'a,'tcx> {
    fn bother(&mut self) -> isize {
        // FIXME (#22405): Replace `Box::new` with `box` here when/if possible.
        self.elaborate_bounds(Box::new(|this| {
            // (*) Here: type of `this` is `&'f0 Foo<&'f1, '_2>`,
            // where `'f0` and `'f1` are fresh, free regions that
            // result from the bound regions on the closure, and `'2`
            // is a region inference variable created by the call. Due
            // to the constraints on the type, we find that `'_2 : 'f1
            // + 'f2` must hold (and can be assumed by the callee).
            // Region inference has to do some clever stuff to avoid
            // inferring `'_2` to be `'static` in this case, because
            // it is created outside the closure but then related to
            // regions bound by the closure itself. See the
            // `region_inference.rs` file (and the `givens` field, in
            // particular) for more details.
            this.foo()
        }))
    }

    fn foo(&mut self) -> isize {
        22
    }

    fn elaborate_bounds(
        &mut self,
        mut mk_cand: Box<for<'b> FnMut(&mut Foo<'b, 'tcx>) -> isize>)
        -> isize
    {
        mk_cand(self)
    }
}

fn main() {
    let v = vec![];
    let cx = Ctxt { x: &v };
    let mut foo = Foo { cx: &cx };
    assert_eq!(foo.bother(), 22); // just so the code is not dead, basically
}
