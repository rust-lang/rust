#![feature(weak_into_raw)]

use std::cell::{Cell, RefCell};
use std::rc::{Rc, Weak};
use std::sync::Arc;
use std::fmt::Debug;

fn rc_refcell() {
    let r = Rc::new(RefCell::new(42));
    let r2 = r.clone();
    *r.borrow_mut() += 10;
    let x = *r2.borrow();
    assert_eq!(x, 52);
}

fn rc_cell() {
    let r = Rc::new(Cell::new(42));
    let r2 = r.clone();
    let x = r.get();
    r2.set(x + x);
    assert_eq!(r.get(), 84);
}

fn rc_refcell2() {
    let r = Rc::new(RefCell::new(42));
    let r2 = r.clone();
    *r.borrow_mut() += 10;
    let x = r2.borrow();
    let r3 = r.clone();
    let y = r3.borrow();
    assert_eq!((*x + *y)/2, 52);
}

fn rc_raw() {
    let r = Rc::new(0);
    let r2 = Rc::into_raw(r.clone());
    let r2 = unsafe { Rc::from_raw(r2) };
    assert!(Rc::ptr_eq(&r, &r2));
    drop(r);
    assert!(Rc::try_unwrap(r2).is_ok());
}

fn arc() {
    fn test() -> Arc<i32> {
        let a = Arc::new(42);
        a
    }
    assert_eq!(*test(), 42);
}

// Make sure this Rc doesn't fall apart when touched
fn check_unique_rc<T: ?Sized>(mut r: Rc<T>) {
    let r2 = r.clone();
    assert!(Rc::get_mut(&mut r).is_none());
    drop(r2);
    assert!(Rc::get_mut(&mut r).is_some());
}

fn rc_from() {
    check_unique_rc::<[_]>(Rc::from(&[1,2,3] as &[_]));
    check_unique_rc::<[_]>(Rc::from(vec![1,2,3]));
    check_unique_rc::<[_]>(Rc::from(Box::new([1,2,3]) as Box<[_]>));
    check_unique_rc::<str>(Rc::from("Hello, World!"));
}

fn rc_fat_ptr_eq() {
    let p = Rc::new(1) as Rc<dyn Debug>;
    let a: *const dyn Debug = &*p;
    let r = Rc::into_raw(p);
    assert!(a == r);
    drop(unsafe { Rc::from_raw(r) });
}

/// Taken from the `Weak::into_raw` doctest.
fn weak_into_raw() {
    let strong = Rc::new(42);
    let weak = Rc::downgrade(&strong);
    let raw = Weak::into_raw(weak);

    assert_eq!(1, Rc::weak_count(&strong));
    assert_eq!(42, unsafe { *raw });

    drop(unsafe { Weak::from_raw(raw) });
    assert_eq!(0, Rc::weak_count(&strong));
}

/// Taken from the `Weak::from_raw` doctest.
fn weak_from_raw() {
    let strong = Rc::new(42);

    let raw_1 = Weak::into_raw(Rc::downgrade(&strong));
    let raw_2 = Weak::into_raw(Rc::downgrade(&strong));

    assert_eq!(2, Rc::weak_count(&strong));

    assert_eq!(42, *Weak::upgrade(&unsafe { Weak::from_raw(raw_1) }).unwrap());
    assert_eq!(1, Rc::weak_count(&strong));

    drop(strong);

    // Decrement the last weak count.
    assert!(Weak::upgrade(&unsafe { Weak::from_raw(raw_2) }).is_none());
}

fn main() {
    rc_fat_ptr_eq();
    rc_refcell();
    rc_refcell2();
    rc_cell();
    rc_raw();
    rc_from();
    weak_into_raw();
    weak_from_raw();

    arc();
}
