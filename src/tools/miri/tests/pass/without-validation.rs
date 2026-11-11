// When we notice something breaks only without validation, we add a test here.
//@compile-flags: -Zmiri-disable-validation
use std::cell::*;

fn refcell_unsize() {
    let cell: RefCell<[i32; 3]> = RefCell::new([1, 2, 3]);
    {
        let mut cellref: RefMut<'_, [i32; 3]> = cell.borrow_mut();
        cellref[0] = 4;
        let mut coerced: RefMut<'_, [i32]> = cellref;
        coerced[2] = 5;
    }
    {
        let comp: &mut [i32] = &mut [4, 2, 5];
        let cellref: Ref<'_, [i32; 3]> = cell.borrow();
        assert_eq!(&*cellref, comp);
        let coerced: Ref<'_, [i32]> = cellref;
        assert_eq!(&*coerced, comp);
    }
}

fn main() {
    refcell_unsize();
}
