// This is a regression test for issues that came up during review of the (closed)
// PR #132289; this 3-ish-crate (including std) test case is adapted from
// the third example from @steffahn during review.
// https://github.com/rust-lang/rust/pull/132289#issuecomment-2564599221

//@ run-pass
//@ check-run-results
//@ aux-build: pr_review_132289_3_lib.rs

extern crate pr_review_132289_3_lib;

use std::ops::Index;

use pr_review_132289_3_lib::{call, Trait};

trait SubIndex<I>: Index<I> {}

struct Param;

trait Project {
    type Ty: ?Sized;
}
impl Project for () {
    type Ty = dyn SubIndex<Param, Output = ()>;
}

impl Index<Param> for <() as Project>::Ty {
    type Output = ();

    fn index(&self, _: Param) -> &() {
        &()
    }
}

struct Struct;

impl Trait for Struct {
    fn f(&self)
    where
        // higher-ranked to allow potentially-false bounds
        for<'a> dyn Index<(), Output = ()>: Index<()>,
        // after #132289 rustc used to believe this bound false
    {
        println!("hello!");
    }
}

fn main() {
    call(&Struct); // <- would segfault if the method `f` wasn't part of the vtable
}
