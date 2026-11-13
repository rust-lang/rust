//@ ignore-compare-mode-polonius (explicit revisions)
//@ edition: 2024
//@ revisions: nll polonius
//@ [nll] compile-flags: -Zpolonius=off
//@ [polonius] compile-flags: -Zpolonius=next

// For an opaque type, we duplicate captured lifetimes (the parent args and the
// opaque's own args). Although the parent args are marked as *bivariant*, in an
// invariant context, those are still considered invariant and participate in
// outlives relationships. Therefore, we must mark them as live.
//
// This specific test demonostrates the path where there *are* captured regions.

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

fn hide<'a, 'b: 'a, T: 'static>(x: Rc<RefCell<&'b T>>) -> impl Swap + use<'a, 'b, T> + 'a {
    x
}

fn dangle() -> &'static [i32; 3] {
    let long = Rc::new(RefCell::new(&[4, 5, 6]));
    let x = [1, 2, 3];
    let short = Rc::new(RefCell::new(&x));
    //~^ ERROR `x` does not live long enough
    hide(long.clone()).swap((|a| a)(hide(short)));
    let res: &'static [i32; 3] = *long.borrow();
    res
}

fn main() {
    println!("{:?}", dangle());
}
