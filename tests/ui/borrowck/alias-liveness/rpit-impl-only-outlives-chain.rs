//@ edition: 2024
//@ ignore-compare-mode-polonius (explicit revisions)
//@ revisions: nll polonius
//@ [nll] compile-flags: -Znext-solver=globally -Zpolonius=off
//@ [polonius] compile-flags: -Znext-solver=globally -Zpolonius=next

// An impl that proves a given where clause may provide extra outlives
// relationships that mean a lifetime can end up in the hidden type, so we are
// careful about where clauses that contain the outlives lifetime and some other
// arg. *However*, as this test demonstrates, we also need to consider transitive
// outliving lifetimes as well.

#![forbid(unsafe_code)]

use std::cell::RefCell;
use std::rc::Rc;

trait Swap: Sized {
    fn swap(self, other: Self);
}
impl<T> Swap for &mut T {
    fn swap(self, other: Self) {
        std::mem::swap(self, other);
    }
}
impl<T> Swap for Rc<RefCell<T>> {
    fn swap(self, other: Self) {
        RefCell::swap(&self, &other);
    }
}

trait Shorten<'m, 'b> {
    fn shorten<T: Swap + 'm>(x: T) -> impl Swap + 'b;
}
impl<'m: 'b, 'b> Shorten<'m, 'b> for () {
    fn shorten<T: Swap + 'm>(x: T) -> impl Swap + 'b {
        x
    }
}

// We treat `'b` as live normally, `'m` as live because `'m` and `'b` appear in
// the where clause together. We must also treat `'a` as live, too, because
// of the explicit `'a: 'm`.
fn hide<'a: 'm, 'm, 'b>(x: Rc<RefCell<&'a [i32; 3]>>) -> impl Swap + 'b
where
    (): Shorten<'m, 'b>,
{
    <() as Shorten<'m, 'b>>::shorten(x)
}

// Still unsound under polonius
fn dangle() -> &'static [i32; 3] {
    let long = Rc::new(RefCell::new(&[4, 5, 6]));
    let x = [1, 2, 3];
    let short = Rc::new(RefCell::new(&x));
    //~^ ERROR `x` does not live long enough
    hide(long.clone()).swap((|a| a)(hide(short)));
    let res: &'static [i32; 3] = *long.borrow();
    res
}

// We must treat `T` as live, because of both the `T: 'm` and the presence of
// both `'m` and `'b` in the where clause.
fn hide_type<'m, 'b, T: Swap + 'm>(x: T) -> impl Swap + 'b
where
    (): Shorten<'m, 'b>,
{
    <() as Shorten<'m, 'b>>::shorten(x)
}

fn dangle_type() -> &'static [i32; 3] {
    let long = Rc::new(RefCell::new(&[4, 5, 6]));
    let x = [1, 2, 3];
    let short = Rc::new(RefCell::new(&x));
    //~^ ERROR `x` does not live long enough
    hide_type(long.clone()).swap((|a| a)(hide_type(short)));
    let res: &'static [i32; 3] = *long.borrow();
    res
}

// We don't need to treat `'c` or `'d` as live because need can outlive '`a`,
// and so cannot be in the hidden type.
fn hide_unrelated<'a, 'c, 'd>(x: &'c mut i32, _: &'d mut i32) -> impl Sized + 'a + use<'a, 'c, 'd>
where
    (): Shorten<'c, 'd>,
{
    // This would error
    //<() as Shorten<'c, 'd>>::shorten(x)
}

fn unrelated_clause() {
    let mut c = 0;
    let mut d = 0;
    let hidden = hide_unrelated(&mut c, &mut d);
    c = 1;
    d = 1;
    drop(hidden);
}


// Although `'c` and `'d` don't occur in a clause together with `'a`, we know
// that `'d: 'a`, so `'d` must be live and it's possible that `'c: 'd` so it must
// be live as well.
fn hide_unrelated_with_outlives<'a, 'c, 'd: 'a>(
    x: &'c mut i32,
    _: &'d mut i32,
) -> impl Sized + 'a + use<'a, 'c, 'd>
where
    (): Shorten<'c, 'd>,
{
    <() as Shorten<'c, 'd>>::shorten(x)
}

fn unrelated_clause_with_outlives() {
    let mut c = 0;
    let mut d = 0;
    let hidden = hide_unrelated_with_outlives(&mut c, &mut d);
    c = 1;
    //~^ ERROR cannot assign to `c` because it is borrowed
    d = 1;
    //~^ ERROR cannot assign to `d` because it is borrowed
    drop(hidden);
}


// Similar to the `dangle` case, but shows a couple other cases where we must
// treat `'a` as live if `'m` is live.
mod projection {
    use super::{Rc, RefCell, Shorten, Swap};

    trait Family {
        type Out<'a>: Swap;
        fn make<'a>(x: Rc<RefCell<&'a [i32; 3]>>) -> Self::Out<'a>;
    }
    impl Family for () {
        type Out<'a> = Rc<RefCell<&'a [i32; 3]>>;
        fn make<'a>(x: Rc<RefCell<&'a [i32; 3]>>) -> Self::Out<'a> {
            x
        }
    }

    fn hide<'a, 'm, 'b, G: Family>(x: Rc<RefCell<&'a [i32; 3]>>) -> impl Swap + 'b
    where
        G::Out<'a>: 'm,
        (): Shorten<'m, 'b>,
    {
        <() as Shorten<'m, 'b>>::shorten(G::make(x))
    }

    fn hide_implied<'a, 'm, 'b, G: Family>(
        x: Rc<RefCell<&'a [i32; 3]>>,
        _: Option<&'m G::Out<'a>>,
    ) -> impl Swap + 'b
    where
        (): Shorten<'m, 'b>,
    {
        <() as Shorten<'m, 'b>>::shorten(G::make(x))
    }

    fn hide_declared<'a, 'm, 'b, G: Family>(
        x: Rc<RefCell<&'a [i32; 3]>>,
    ) -> impl Swap + 'b
    where
        G::Out<'a>: 'm,
        'm: 'b,
    {
        G::make(x)
    }

    // Still unsound under polonius
    fn dangle() -> &'static [i32; 3] {
        let long = Rc::new(RefCell::new(&[4, 5, 6]));
        let x = [1, 2, 3];
        let short = Rc::new(RefCell::new(&x));
        //~^ ERROR `x` does not live long enough
        hide::<()>(long.clone()).swap((|a| a)(hide::<()>(short)));
        let res: &'static [i32; 3] = *long.borrow();
        res
    }

    fn dangle_implied() -> &'static [i32; 3] {
        let long = Rc::new(RefCell::new(&[4, 5, 6]));
        let x = [1, 2, 3];
        let short = Rc::new(RefCell::new(&x));
        //~^ ERROR `x` does not live long enough
        hide_implied::<()>(long.clone(), None).swap((|a| a)(hide_implied::<()>(short, None)));
        let res: &'static [i32; 3] = *long.borrow();
        res
    }

    // Still unsound under polonius
    fn dangle_declared() -> &'static [i32; 3] {
        let long = Rc::new(RefCell::new(&[4, 5, 6]));
        let x = [1, 2, 3];
        let short = Rc::new(RefCell::new(&x));
        //~^ ERROR `x` does not live long enough
        hide_declared::<()>(long.clone()).swap((|a| a)(hide_declared::<()>(short)));
        let res: &'static [i32; 3] = *long.borrow();
        res
    }

    pub fn check() {
        println!("{:?}", dangle());
        println!("{:?}", dangle_implied());
        println!("{:?}", dangle_declared());
    }
}

fn main() {
    println!("{:?}", dangle());
    println!("{:?}", dangle_type());
    unrelated_clause();
    projection::check();
}
