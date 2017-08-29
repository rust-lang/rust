use std::cell::RefCell;
use std::rc::Rc;

fn rc_refcell() {
    let r = Rc::new(RefCell::new(42));
    *r.borrow_mut() += 10;
    let x = *r.borrow();
    assert_eq!(x, 52);
}

fn rc_raw() {
    let r = Rc::new(0);
    let r2 = Rc::into_raw(r.clone());
    let r2 = unsafe { Rc::from_raw(r2) };
    assert!(Rc::ptr_eq(&r, &r2));
    drop(r);
    assert!(Rc::try_unwrap(r2).is_ok());
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

fn main() {
    rc_refcell();
    rc_raw();
    rc_from();
}
