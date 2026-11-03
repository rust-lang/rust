//@ ignore-compare-mode-polonius (explicit revisions)
//@ edition: 2024
//@ revisions: nll polonius
//@ [nll] compile-flags: -Zpolonius=off
//@ [polonius] compile-flags: -Zpolonius=next
//@ [polonius] check-pass
//@ [polonius] known-bug: #153215

// Data available through an associated type can end up in the hidden type of an
// opaque. We currently don't treat any generics as live, although we *should*.

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

fn hide_one_bound<'a, S: Tr>(x: S) -> impl Swap + 'a
where
    S::Assoc: 'a,
{
    x.get()
}

fn dangle_one_bound() -> &'static [i32; 3] {
    let long = Rc::new(RefCell::new(&[4, 5, 6] as &'static [i32; 3]));
    let x = [1, 2, 3];
    let short = Rc::new(RefCell::new(&x));
    //[nll]~^ ERROR `x` does not live long enough
    hide_one_bound(long.clone()).swap(hide_one_bound(short));
    let res: &'static [i32; 3] = *long.borrow();
    res
}

 trait TrItemBound<'a> {
    type Assoc: Swap + 'a;
    fn get(self) -> Self::Assoc;
}

impl<'q, 'a> TrItemBound<'a> for Rc<RefCell<&'q [i32; 3]>>
where
    'q: 'a,
{
    type Assoc = Rc<RefCell<&'q [i32; 3]>>;
    fn get(self) -> Self::Assoc {
        self
    }
}

fn hide_item_bound<'a, S: TrItemBound<'a>>(x: S) -> impl Swap + 'a {
    x.get()
}

fn dangle_item_bound() -> &'static [i32; 3] {
    let long = Rc::new(RefCell::new(&[4, 5, 6] as &'static [i32; 3]));
    let x = [1, 2, 3];
    let short = Rc::new(RefCell::new(&x));
    //[nll]~^ ERROR `x` does not live long enough
    hide_item_bound(long.clone()).swap(hide_item_bound(short));
    let res: &'static [i32; 3] = *long.borrow();
    res
}

fn main() {
    println!("{:?}", dangle_one_bound());
    println!("{:?}", dangle_item_bound());
}
