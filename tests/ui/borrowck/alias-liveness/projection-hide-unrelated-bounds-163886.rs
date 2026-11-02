//@ ignore-compare-mode-polonius (explicit revisions)
//@ edition: 2024
//@ revisions: nll polonius
//@ [nll] compile-flags: -Zpolonius=off
//@ [polonius] compile-flags: -Zpolonius=next
//@ [polonius] check-pass
//@ [polonius] known-bug: #163886

// Similar to `projection-hide-one-bound-153215.rs`, but with two unrelated
// bounds. This used to properly error with Polonius Alpha, but has regressed
// since #156027.

#![forbid(unsafe_code)]

use std::cell::RefCell;
use std::rc::Rc;

trait Swap: Sized {
    fn swap(self, other: Self);
}

impl<T> Swap for Rc<RefCell<T>> {
    fn swap(self, other: Self) {
        <RefCell<T>>::swap(&self, &other);
    }
}

trait Tr {
    type Assoc: Swap;
    fn get(self) -> Self::Assoc;
}

impl<'q> Tr for Rc<RefCell<&'q [i32; 3]>> {
    type Assoc = Rc<RefCell<&'q [i32; 3]>>;
    fn get(self) -> Self::Assoc {
        self
    }
}

fn hide_unrelated_bounds<'a, 'b, S: Tr>(x: S) -> impl Swap + 'a + 'b
where
    S::Assoc: 'a + 'b,
{
    x.get()
}

fn dangle_unrelated_bounds() -> &'static [i32; 3] {
    let long = Rc::new(RefCell::new(&[4, 5, 6] as &'static [i32; 3]));
    let x = [1, 2, 3];
    let short = Rc::new(RefCell::new(&x));
    //[nll]~^ ERROR `x` does not live long enough
    hide_unrelated_bounds(long.clone()).swap(hide_unrelated_bounds(short));
    let res: &'static [i32; 3] = *long.borrow();
    res
}

fn main() {
    println!("{:?}", dangle_unrelated_bounds());
}
