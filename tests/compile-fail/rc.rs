//error-pattern: no mir for `std::result::unwrap_failed::__STATIC_FMTSTR`

use std::cell::RefCell;
use std::rc::Rc;

fn rc_refcell() -> i32 {
    let r = Rc::new(RefCell::new(42));
    *r.borrow_mut() += 10;
    let x = *r.borrow();
    x
}

fn main() {
    rc_refcell();
}
