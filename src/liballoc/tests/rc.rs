use std::any::Any;
use std::rc::{Rc, Weak};

#[test]
fn uninhabited() {
    enum Void {}
    let mut a = Weak::<Void>::new();
    a = a.clone();
    assert!(a.upgrade().is_none());

    let mut a: Weak<dyn Any> = a;  // Unsizing
    a = a.clone();
    assert!(a.upgrade().is_none());
}

#[test]
fn slice() {
    let a: Rc<[u32; 3]> = Rc::new([3, 2, 1]);
    let a: Rc<[u32]> = a;  // Unsizing
    let b: Rc<[u32]> = Rc::from(&[3, 2, 1][..]);  // Conversion
    assert_eq!(a, b);

    // Exercise is_dangling() with a DST
    let mut a = Rc::downgrade(&a);
    a = a.clone();
    assert!(a.upgrade().is_some());
}

#[test]
fn trait_object() {
    let a: Rc<u32> = Rc::new(4);
    let a: Rc<dyn Any> = a;  // Unsizing

    // Exercise is_dangling() with a DST
    let mut a = Rc::downgrade(&a);
    a = a.clone();
    assert!(a.upgrade().is_some());

    let mut b = Weak::<u32>::new();
    b = b.clone();
    assert!(b.upgrade().is_none());
    let mut b: Weak<dyn Any> = b;  // Unsizing
    b = b.clone();
    assert!(b.upgrade().is_none());
}
