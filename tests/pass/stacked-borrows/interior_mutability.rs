// compile-flags: -Zmiri-tag-raw-pointers
use std::cell::{Cell, RefCell, UnsafeCell};
use std::mem::{self, MaybeUninit};

fn main() {
    aliasing_mut_and_shr();
    aliasing_frz_and_shr();
    into_interior_mutability();
    unsafe_cell_2phase();
    unsafe_cell_deallocate();
    unsafe_cell_invalidate();
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
    assert_eq!(*rc.borrow(), 23 + 12);
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
fn unsafe_cell_2phase() {
    unsafe {
        let x = &UnsafeCell::new(vec![]);
        let x2 = &*x;
        (*x.get()).push(0);
        let _val = (*x2.get()).get(0);
    }
}

/// Make sure we can deallocate an UnsafeCell that was passed to an active fn call.
/// (This is the fix for https://github.com/rust-lang/rust/issues/55005.)
fn unsafe_cell_deallocate() {
    fn f(x: &UnsafeCell<i32>) {
        let b: Box<i32> = unsafe { Box::from_raw(x as *const _ as *mut i32) };
        drop(b)
    }

    let b = Box::new(0i32);
    f(unsafe { mem::transmute(Box::into_raw(b)) });
}

/// As a side-effect of the above, we also allow this -- at least for now.
fn unsafe_cell_invalidate() {
    fn f(_x: &UnsafeCell<i32>, y: *mut i32) {
        // Writing to y invalidates x, but that is okay.
        unsafe {
            *y += 1;
        }
    }

    let mut x = 0i32;
    let raw1 = &mut x as *mut _;
    let ref1 = unsafe { &mut *raw1 };
    let raw2 = ref1 as *mut _;
    // Now the borrow stack is: raw1, ref2, raw2.
    // So using raw1 invalidates raw2.
    f(unsafe { mem::transmute(raw2) }, raw1);
}
