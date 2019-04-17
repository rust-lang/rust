use std::cell::RefCell;

fn main() {
    aliasing_mut_and_shr();
    aliasing_frz_and_shr();
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
