use std::mem::MaybeUninit;
use std::cell::{Cell, RefCell, UnsafeCell};

fn main() {
    aliasing_mut_and_shr();
    aliasing_frz_and_shr();
    into_interior_mutability();
    unsafe_cell_2phase();
}

fn aliasing_mut_and_shr() {
    fn inner(rc: &RefCell<i32>, aliasing: &mut i32) {
        *aliasing += 4;
        let _escape_to_raw = rc as *const _;
        *aliasing += 4;
        let _shr = &*rc;
        *aliasing += 4;
        // also turning this into a frozen ref now must work
        let aliasing = &*aliasing;
        let _val = *aliasing;
        let _escape_to_raw = rc as *const _; // this must NOT unfreeze
        let _val = *aliasing;
        let _shr = &*rc; // this must NOT unfreeze
        let _val = *aliasing;
    }

    let rc = RefCell::new(23);
    let mut bmut = rc.borrow_mut();
    inner(&rc, &mut *bmut);
    drop(bmut);
    assert_eq!(*rc.borrow(), 23+12);
}

fn aliasing_frz_and_shr() {
    fn inner(rc: &RefCell<i32>, aliasing: &i32) {
        let _val = *aliasing;
        let _escape_to_raw = rc as *const _; // this must NOT unfreeze
        let _val = *aliasing;
        let _shr = &*rc; // this must NOT unfreeze
        let _val = *aliasing;
    }

    let rc = RefCell::new(23);
    let bshr = rc.borrow();
    inner(&rc, &*bshr);
    assert_eq!(*rc.borrow(), 23);
}

// Getting a pointer into a union with interior mutability used to be tricky
// business (https://github.com/rust-lang/miri/issues/615), but it should work
// now.
fn into_interior_mutability() {
    let mut x: MaybeUninit<(Cell<u32>, u32)> = MaybeUninit::uninit();
    x.as_ptr();
    x.write((Cell::new(0), 1));
    let ptr = unsafe { x.assume_init_ref() };
    assert_eq!(ptr.1, 1);
}

// Two-phase borrows of the pointer returned by UnsafeCell::get() should not
// invalidate aliases.
fn unsafe_cell_2phase() { unsafe {
    let x = &UnsafeCell::new(vec![]);
    let x2 = &*x;
    (*x.get()).push(0);
    let _val = (*x2.get()).get(0);
} }
