//@ ignore-compare-mode-polonius (explicit revisions)
//@ edition: 2024
//@ revisions: nll polonius
//@ [nll] compile-flags: -Zpolonius=off
//@ [polonius] compile-flags: -Zpolonius=next

// Data available through an associated type can end up in the hidden type of an opaque.

#![forbid(unsafe_code)]

use std::cell::RefCell;
use std::marker::PhantomData;
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

// The outlives clause names `S::Assoc` directly, rather than `S`.
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
    //~^ ERROR `x` does not live long enough
    hide_one_bound(long.clone()).swap(hide_one_bound(short));
    let res: &'static [i32; 3] = *long.borrow();
    res
}

// The trait's associated type bound supplies `S::Assoc: 'a`.
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
    //~^ ERROR `x` does not live long enough
    hide_item_bound(long.clone()).swap(hide_item_bound(short));
    let res: &'static [i32; 3] = *long.borrow();
    res
}

trait Inner<'a> {
    type Out: Swap + 'a;
    fn conv(self) -> Self::Out;
}

impl<'a, 'q: 'a> Inner<'a> for Rc<RefCell<&'q [i32; 3]>> {
    type Out = Rc<RefCell<&'q [i32; 3]>>;
    fn conv(self) -> Self::Out {
        self
    }
}

// `S::Assoc: Inner<'a>` gives `<S::Assoc as Inner<'a>>::Out: 'a
fn hide_trait_bound_on_projection<'a, S: Tr>(x: S) -> impl Swap + 'a
where
    S::Assoc: Inner<'a>,
{
    x.get().conv()
}

fn dangle_trait_bound_on_projection() -> &'static [i32; 3] {
    let long = Rc::new(RefCell::new(&[4, 5, 6] as &'static [i32; 3]));
    let x = [1, 2, 3];
    let short = Rc::new(RefCell::new(&x));
    //~^ ERROR `x` does not live long enough
    hide_trait_bound_on_projection(long.clone()).swap(hide_trait_bound_on_projection(short));
    let res: &'static [i32; 3] = *long.borrow();
    res
}

struct Holder<'a, S: Tr>(PhantomData<(&'a (), S)>)
where
    S::Assoc: 'a;

// WF of the `Holder<'a, S>` argument implies `S::Assoc: 'a`
fn hide_implied_bound_from_adt<'a, S: Tr>(x: S, _h: Holder<'a, S>) -> impl Swap + 'a {
    x.get()
}

fn dangle_implied_bound_from_adt() -> &'static [i32; 3] {
    let long = Rc::new(RefCell::new(&[4, 5, 6] as &'static [i32; 3]));
    let x = [1, 2, 3];
    let short = Rc::new(RefCell::new(&x));
    //~^ ERROR `x` does not live long enough
    hide_implied_bound_from_adt(long.clone(), Holder(PhantomData))
        .swap(hide_implied_bound_from_adt(short, Holder(PhantomData)));
    let res: &'static [i32; 3] = *long.borrow();
    res
}

fn main() {
    println!("{:?}", dangle_one_bound());
    println!("{:?}", dangle_item_bound());
    println!("{:?}", dangle_trait_bound_on_projection());
    println!("{:?}", dangle_implied_bound_from_adt());
}
