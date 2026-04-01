// This should fail even without validation
//@compile-flags: -Zmiri-disable-validation

use std::ptr;
use std::rc::{Rc, Weak};

/// Taken from the `Weak::as_ptr` doctest.
fn main() {
    let strong = Rc::new(Box::new(42));
    let weak = Rc::downgrade(&strong);
    // Both point to the same object
    assert!(ptr::eq(&*strong, Weak::as_ptr(&weak)));
    // The strong here keeps it alive, so we can still access the object.
    assert_eq!(42, **unsafe { &*Weak::as_ptr(&weak) });

    drop(strong);
    // But not any more. We can do Weak::as_raw(&weak), but accessing the pointer would lead to
    // undefined behaviour.
    assert_eq!(42, **unsafe { &*Weak::as_ptr(&weak) }); //~ ERROR: has been freed
}
