// run-pass

//! Test that let bindings and destructuring assignments have consistent drop orders

#![feature(destructuring_assignment)]
#![allow(unused_variables, unused_assignments)]

use std::cell::RefCell;

thread_local! {
    static DROP_ORDER: RefCell<Vec<usize>> = RefCell::new(Vec::new());
}

struct DropRecorder(usize);
impl Drop for DropRecorder {
    fn drop(&mut self) {
        DROP_ORDER.with(|d| d.borrow_mut().push(self.0));
    }
}

fn main() {
    let expected_drop_order = vec![1, 4, 5, 3, 2];
    // Check the drop order for let bindings:
    {
        let _ = DropRecorder(1);
        let _val = DropRecorder(2);
        let (x, _) = (DropRecorder(3), DropRecorder(4));
        drop(DropRecorder(5));
    }
    DROP_ORDER.with(|d| {
        assert_eq!(&*d.borrow(), &expected_drop_order);
        d.borrow_mut().clear();
    });
    // Check that the drop order for destructuring assignment is the same:
    {
        let _val;
        let x;
        _ = DropRecorder(1);
        _val = DropRecorder(2);
        (x, _) = (DropRecorder(3), DropRecorder(4));
        drop(DropRecorder(5));
    }
    DROP_ORDER.with(|d| assert_eq!(&*d.borrow(), &expected_drop_order));
}
