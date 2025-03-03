//@ run-pass
//! This tests `cell::Ref` and `cell::RefMut` are usable in `const` contexts

#![feature(const_ref_cell)]
#![feature(const_destruct)]

use std::mem;
use std::sync::atomic::{AtomicU8, Ordering};
use std::cell::RefCell;

static DROP_COUNT: AtomicU8 = AtomicU8::new(0);

struct Dummy;

impl Drop for Dummy {
    fn drop(&mut self){
        DROP_COUNT.fetch_add(1, Ordering::Relaxed);
    }
}

const REF_CELL_TEST: RefCell<Dummy> = {
    let a = RefCell::new(Dummy);

    // Check that `borrow` guards correctly at compile-time
    {
        assert!(a.try_borrow().is_ok());
        assert!(a.try_borrow_mut().is_ok());
        let _a = a.borrow();
        assert!(a.try_borrow().is_ok());
        assert!(a.try_borrow_mut().is_err());
    }

    // Check that `borrow_mut` guards correctly at compile-time
    {
        assert!(a.try_borrow().is_ok());
        assert!(a.try_borrow_mut().is_ok());
        let _a = a.borrow_mut();
        assert!(a.try_borrow().is_err());
        assert!(a.try_borrow_mut().is_err());
    }

    a
};

fn main() {
    let dummy = REF_CELL_TEST;
    assert_eq!(DROP_COUNT.load(Ordering::Relaxed), 0);
    mem::forget(dummy);
}
