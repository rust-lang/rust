// Test to show what happens if we were not careful and allowed invariant
// lifetimes to escape though an impl trait.
//
// Specifically we swap a long lived and short lived reference, giving us a
// dangling pointer.

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
        <RefCell<T>>::swap(&self, &other);
    }
}

// Here we are hiding `'b` making the caller believe that `&'a mut &'s T` and
// `&'a mut &'l T` are the same type.
fn hide_ref<'a, 'b, T: 'static>(x: &'a mut &'b T) -> impl Swap + 'a {
    //~^ ERROR hidden type
    x
}

fn dangle_ref() -> &'static [i32; 3] {
    let mut res = &[4, 5, 6];
    let x = [1, 2, 3];
    hide_ref(&mut res).swap(hide_ref(&mut &x));
    res
}

// Here we are hiding `'b` making the caller believe that `Rc<RefCell<&'s T>>`
// and `Rc<RefCell<&'l T>>` are the same type.
//
// This is different to the previous example because the concrete return type
// only has a single lifetime.
fn hide_rc_refcell<'a, 'b: 'a, T: 'static>(x: Rc<RefCell<&'b T>>) -> impl Swap + 'a {
    //~^ ERROR hidden type
    x
}

fn dangle_rc_refcell() -> &'static [i32; 3] {
    let long = Rc::new(RefCell::new(&[4, 5, 6]));
    let x = [1, 2, 3];
    let short = Rc::new(RefCell::new(&x));
    hide_rc_refcell(long.clone()).swap(hide_rc_refcell(short));
    let res: &'static [i32; 3] = *long.borrow();
    res
}

fn main() {
    // both will print nonsense values.
    println!("{:?}", dangle_ref());
    println!("{:?}", dangle_rc_refcell())
}
