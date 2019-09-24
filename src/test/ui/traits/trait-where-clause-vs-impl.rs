// run-pass
#![allow(dead_code)]
#![allow(unused_variables)]
// Test that when there is a conditional (but blanket) impl and a
// where clause, we don't get confused in trait resolution.
//
// Issue #18453.

// pretty-expanded FIXME #23616

use std::rc::Rc;

pub trait Foo<M> {
    fn foo(&mut self, msg: M);
}

pub trait Bar<M> {
    fn dummy(&self) -> M;
}

impl<M, F: Bar<M>> Foo<M> for F {
    fn foo(&mut self, msg: M) {
    }
}

pub struct Both<M, F> {
    inner: Rc<(M, F)>,
}

impl<M, F: Foo<M>> Clone for Both<M, F> {
    fn clone(&self) -> Both<M, F> {
        Both { inner: self.inner.clone() }
    }
}

fn repro1<M, F: Foo<M>>(_both: Both<M, F>) {
}

fn repro2<M, F: Foo<M>>(msg: M, foo: F) {
    let both = Both { inner: Rc::new((msg, foo)) };
    repro1(both.clone()); // <--- This clone causes problem
}

pub fn main() {
}
