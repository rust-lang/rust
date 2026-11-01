//@ ignore-compare-mode-next-solver (explicit revisions)
//@ ignore-compare-mode-polonius (explicit revisions)
//@ edition: 2024
//@ revisions: nll_current nll_next polonius_current polonius_next
//@ [nll_current] compile-flags: -Zpolonius=off
//@ [nll_next] compile-flags: -Znext-solver=globally -Zpolonius=off
//@ [polonius_current] compile-flags: -Zpolonius=next
//@ [polonius_next] compile-flags: -Znext-solver=globally -Zpolonius=next

// `Equal<'a, 'b>` has implied `'a: 'b` and `'b: 'a` bounds, therefore `make`
// and `make2` should count `T` as live. However, because of #106569, we need
// to be careful when there is a projection where normalization equates regions.

#![forbid(unsafe_code)]

use std::cell::RefCell;
use std::rc::Rc;

trait Swap: Sized {
    fn swap(self, other: Self);
}

impl<T> Swap for Rc<RefCell<T>> {
    fn swap(self, other: Self) {
        RefCell::swap(&self, &other);
    }
}

struct Equal<'a, 'b>(&'a &'b (), &'b &'a ());

trait Trait {
    type Ty;
}

impl<'x> Trait for Equal<'x, 'x> {
    type Ty = ();
}

// This is a problem under #106569.
fn make<'a: 'a, 'b: 'b, T: Swap + 'a>(
    //[nll_current,polonius_current]~^ ERROR lifetime may not live long enough
    //[nll_current,polonius_current]~| ERROR lifetime may not live long enough
    //[nll_current,polonius_current]~| ERROR one or more lifetime errors were found in this item
    _: (<Equal<'a, 'b> as Trait>::Ty, Equal<'a, 'b>),
    t: T,
) -> impl Swap + 'b {
    t
}

// This is a not a problem under #106569.
fn make2<'a: 'a, 'b: 'b, T: Swap + 'a>(_: Equal<'a, 'b>, t: T) -> impl Swap + 'b {
    t
}

fn check() -> &'static u8 {
    let long = Rc::new(RefCell::new(&42u8 as &'static u8));
    let value = 43u8;
    let short = Rc::new(RefCell::new(&value));
    //[nll_current,nll_next,polonius_current,polonius_next]~^ ERROR `value` does not live long enough

    make(((), Equal(&&(), &&())), long.clone()).swap(make(((), Equal(&&(), &&())), short));

    let result: &'static u8 = *long.borrow();
    result
}

fn check2() -> &'static u8 {
    let long = Rc::new(RefCell::new(&42u8 as &'static u8));
    let value = 43u8;
    let short = Rc::new(RefCell::new(&value));
    //[nll_current,nll_next,polonius_current,polonius_next]~^ ERROR `value` does not live long enough

    make2(Equal(&&(), &&()), long.clone()).swap(make2(Equal(&&(), &&()), short));

    let result: &'static u8 = *long.borrow();
    result
}

fn main() {}
