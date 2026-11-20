//! Regression test for <https://github.com/rust-lang/rust/issues/53156>.
//! Second where bound was causing incorrect rejection of acceptable lifetime.

//@ check-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

use std::borrow::Borrow;
use std::sync::Arc;

struct OwnedWrap<B: ToOwned + ?Sized>(<B as ToOwned>::Owned);

impl<'a, B> OwnedWrap<B>
where
    B: 'a + ToOwned + ?Sized,
    &'a B: Into<Arc<B>>,
{
    pub fn to_arc_a(&self) -> Arc<B> {
        let OwnedWrap(ref owned) = *self;
        let b: &B = owned.borrow();
        unimplemented!()
    }

    pub fn to_arc_b(&self) -> Arc<B> {
        let OwnedWrap(ref owned) = *self;
        let _: &B = owned.borrow();
        unimplemented!()
    }
}

fn to_arc_a<'a, B>(wrap: &OwnedWrap<B>) -> Arc<B>
where
    B: 'a + ToOwned + ?Sized,
    &'a B: Into<Arc<B>>,
{
    let OwnedWrap(ref owned) = wrap;
    let b: &B = owned.borrow();
    unimplemented!()
}

fn to_arc_b<'a, B>(wrap: &OwnedWrap<B>) -> Arc<B>
where
    B: 'a + ToOwned + ?Sized,
    &'a B: Into<Arc<B>>,
{
    let OwnedWrap(ref owned) = wrap;
    let _: &B = owned.borrow();
    unimplemented!()
}

fn main() {}
