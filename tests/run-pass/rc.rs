use std::cell::RefCell;
use std::rc::Rc;

fn rc_refcell() -> i32 {
    let r = Rc::new(RefCell::new(42));
    *r.borrow_mut() += 10;
    let x = *r.borrow();
    x
}

fn rc_raw() {
    let r = Rc::new(0);
    let r2 = Rc::into_raw(r.clone());
    let r2 = unsafe { Rc::from_raw(r2) };
    assert!(Rc::ptr_eq(&r, &r2));
    drop(r);
    assert!(Rc::try_unwrap(r2).is_ok());
}

fn main() {
    rc_refcell();
    rc_raw();
}
