//! Regression test for <https://github.com/rust-lang/rust/issues/53156>.
//! Second where bound was causing incorrect rejection of acceptable lifetime.

//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

use std::borrow::Borrow;
use std::sync::Arc;

struct OwnedWrap<B: ToOwned + ?Sized>(<B as ToOwned>::Owned);

impl<'a, B> OwnedWrap<B>
where
    B: 'a + ToOwned + ?Sized,
    &'a B: Into<Arc<B>>, // 1a
{
    // 1a-2a-3a
    pub fn to_arc_aaa(&self) -> Arc<B> {
        let OwnedWrap(ref owned) = *self;
        let b: &B = owned.borrow();
        unimplemented!()
    }

    // 1a-2a-3b
    pub fn to_arc_aab(&self) -> Arc<B> {
        let OwnedWrap(ref owned) = *self;
        let _: &B = owned.borrow();
        unimplemented!()
    }

    // 1a-2a-3c
    pub fn to_arc_aac(&self) -> Arc<B> {
        let OwnedWrap(ref owned) = *self;
        owned.borrow().into()
        //~^ ERROR lifetime may not live long enough
    }

    // 1a-2b-3a
    pub fn to_arc_aba(&'a self) -> Arc<B> {
        let OwnedWrap(ref owned) = *self;
        let b: &B = owned.borrow();
        unimplemented!()
    }

    // 1a-2b-3b
    pub fn to_arc_abb(&'a self) -> Arc<B> {
        let OwnedWrap(ref owned) = *self;
        let _: &B = owned.borrow();
        unimplemented!()
    }

    // 1a-2b-3c
    pub fn to_arc_abc(&'a self) -> Arc<B> {
        let OwnedWrap(ref owned) = *self;
        owned.borrow().into()
    }
}

impl<'a, B> OwnedWrap<B>
where
    B: 'a + ToOwned + ?Sized,
    for<'b> &'b B: Into<Arc<B>> // 1b
{
    // 1b-2a-3a
    pub fn to_arc_baa(&self) -> Arc<B> {
        let OwnedWrap(ref owned) = *self;
        let b: &B = owned.borrow();
        unimplemented!()
    }

    // 1b-2a-3b
    pub fn to_arc_bab(&self) -> Arc<B> {
        let OwnedWrap(ref owned) = *self;
        let _: &B = owned.borrow();
        unimplemented!()
    }

    // 1b-2a-3c
    pub fn to_arc_bac(&self) -> Arc<B> {
        let OwnedWrap(ref owned) = *self;
        owned.borrow().into()
    }

    // 1b-2b-3a
    pub fn to_arc_bba(&'a self) -> Arc<B> {
        let OwnedWrap(ref owned) = *self;
        let b: &B = owned.borrow();
        unimplemented!()
    }

    // 1b-2b-3b
    pub fn to_arc_bbb(&'a self) -> Arc<B> {
        let OwnedWrap(ref owned) = *self;
        let _: &B = owned.borrow();
        unimplemented!()
    }

    // 1b-2b-3c
    pub fn to_arc_bbc(&'a self) -> Arc<B> {
        let OwnedWrap(ref owned) = *self;
        owned.borrow().into()
    }
}

// 1a-2a-3a
fn to_arc_aaa<'a, B>(wrap: &OwnedWrap<B>) -> Arc<B>
where
    B: 'a + ToOwned + ?Sized,
    &'a B: Into<Arc<B>>, // 1a
{
    let OwnedWrap(ref owned) = *wrap;
    let b: &B = owned.borrow();
    unimplemented!()
}

// 1a-2a-3b
fn to_arc_aab<'a, B>(wrap: &OwnedWrap<B>) -> Arc<B>
where
    B: 'a + ToOwned + ?Sized,
    &'a B: Into<Arc<B>>, // 1a
{
    let OwnedWrap(ref owned) = *wrap;
    let _: &B = owned.borrow();
    unimplemented!()
}

// 1a-2a-3c
fn to_arc_aac<'a, B>(wrap: &OwnedWrap<B>) -> Arc<B>
where
    B: 'a + ToOwned + ?Sized,
    &'a B: Into<Arc<B>>, // 1a
{
    let OwnedWrap(ref owned) = *wrap;
    owned.borrow().into()
    //~^ ERROR explicit lifetime required in the type of `wrap`
}

// 1a-2b-3a
fn to_arc_aba<'a, B>(wrap: &'a OwnedWrap<B>) -> Arc<B>
where
    B: 'a + ToOwned + ?Sized,
    &'a B: Into<Arc<B>>, // 1a
{
    let OwnedWrap(ref owned) = *wrap;
    let b: &B = owned.borrow();
    unimplemented!()
}

// 1a-2b-3b
fn to_arc_abb<'a, B>(wrap: &'a OwnedWrap<B>) -> Arc<B>
where
    B: 'a + ToOwned + ?Sized,
    &'a B: Into<Arc<B>>, // 1a
{
    let OwnedWrap(ref owned) = *wrap;
    let _: &B = owned.borrow();
    unimplemented!()
}

// 1a-2b-3c
fn to_arc_abc<'a, B>(wrap: &'a OwnedWrap<B>) -> Arc<B>
where
    B: 'a + ToOwned + ?Sized,
    &'a B: Into<Arc<B>>, // 1a
{
    let OwnedWrap(ref owned) = *wrap;
    owned.borrow().into()
}

// 1b-2a-3a
fn to_arc_baa<'a, B>(wrap: &OwnedWrap<B>) -> Arc<B>
where
    B: 'a + ToOwned + ?Sized,
    for<'b> &'b B: Into<Arc<B>> // 1b
{
    let OwnedWrap(ref owned) = *wrap;
    let b: &B = owned.borrow();
    unimplemented!()
}

// 1b-2a-3b
fn to_arc_bab<'a, B>(wrap: &OwnedWrap<B>) -> Arc<B>
where
    B: 'a + ToOwned + ?Sized,
    for<'b> &'b B: Into<Arc<B>> // 1b
{
    let OwnedWrap(ref owned) = *wrap;
    let _: &B = owned.borrow();
    unimplemented!()
}

// 1b-2a-3c
fn to_arc_bac<'a, B>(wrap: &OwnedWrap<B>) -> Arc<B>
where
    B: 'a + ToOwned + ?Sized,
    for<'b> &'b B: Into<Arc<B>> // 1b
{
    let OwnedWrap(ref owned) = *wrap;
    owned.borrow().into()
}

// 1b-2b-3a
fn to_arc_bba<'a, B>(wrap: &'a OwnedWrap<B>) -> Arc<B>
where
    B: 'a + ToOwned + ?Sized,
    for<'b> &'b B: Into<Arc<B>> // 1b
{
    let OwnedWrap(ref owned) = *wrap;
    let b: &B = owned.borrow();
    unimplemented!()
}

// 1b-2b-3b
fn to_arc_bbb<'a, B>(wrap: &'a OwnedWrap<B>) -> Arc<B>
where
    B: 'a + ToOwned + ?Sized,
    for<'b> &'b B: Into<Arc<B>> // 1b
{
    let OwnedWrap(ref owned) = *wrap;
    let _: &B = owned.borrow();
    unimplemented!()
}

// 1b-2b-3c
fn to_arc_bbc<'a, B>(wrap: &'a OwnedWrap<B>) -> Arc<B>
where
    B: 'a + ToOwned + ?Sized,
    for<'b> &'b B: Into<Arc<B>> // 1b
{
    let OwnedWrap(ref owned) = *wrap;
    owned.borrow().into()
}

fn main() {}
