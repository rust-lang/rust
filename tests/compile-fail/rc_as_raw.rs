// This should fail even without validation
// compile-flags: -Zmiri-disable-validation
#![feature(weak_into_raw)]

use std::rc::{Rc, Weak};
use std::ptr;

/// Taken from the `Weak::as_raw` doctest.
fn main() {
    let strong = Rc::new(Box::new(42));
    let weak = Rc::downgrade(&strong);
    // Both point to the same object
    assert!(ptr::eq(&*strong, Weak::as_raw(&weak)));
    // The strong here keeps it alive, so we can still access the object.
    assert_eq!(42, **unsafe { &*Weak::as_raw(&weak) });
    
    drop(strong);
    // But not any more. We can do Weak::as_raw(&weak), but accessing the pointer would lead to
    // undefined behaviour.
    assert_eq!(42, **unsafe { &*Weak::as_raw(&weak) }); //~ ERROR dangling pointer
}
