//@ edition: 2024
//@ ignore-compare-mode-polonius (explicit revisions)
//@ revisions: nll polonius
//@ [nll] compile-flags: -Znext-solver=globally -Zassumptions-on-binders -Zpolonius=off
//@ [polonius] compile-flags: -Znext-solver=globally -Zassumptions-on-binders -Zpolonius=next

// Unlike `specialized-projection-hidden-lifetime.rs`, this test doesn't have a
// where clause that allows us to easily mark `P` as live.

#![feature(specialization)]
#![allow(incomplete_features)]
#![forbid(unsafe_code)]

use std::fmt::Debug;

trait Select<'a, 'b> {
    type Out: Debug + 'a + 'b;
    fn make(self) -> Self::Out;
}

impl<'a, 'b, P> Select<'a, 'b> for P {
    default type Out = ();
    default fn make(self) -> Self::Out {
        loop {}
    }
}

struct Carrier<'c>(&'c u8);

impl<'a, 'b, 'c: 'a + 'b> Select<'a, 'b> for Carrier<'c> {
    type Out = &'c u8;
    fn make(self) -> Self::Out {
        self.0
    }
}

fn hide<'a, 'b, P>(p: P) -> impl Debug + 'a + 'b {
    //~^ ERROR one or more lifetime errors were found in this item
    <P as Select<'a, 'b>>::make(p)
    //~^ ERROR lifetime may not live long enough
    //~| ERROR lifetime may not live long enough
}

fn control<'a, 'b, P>(_: P) -> impl Debug + 'a + 'b {
    ()
}

fn dangle() {
    let escaped;
    {
        let v = vec![42];
        escaped = hide(Carrier(&v[0]));
    }
    let reused = vec![7];
    println!("escaped = {escaped:?}, reused = {reused:?}");
}

fn main() {
    dangle();
}
