use std::cell::{RefCell, Ref, RefMut};

fn main() {
    basic();
    ref_protector();
    ref_mut_protector();
    rust_issue_68303();
}

fn basic() {
    let c = RefCell::new(42);
    {
        let s1 = c.borrow();
        let _x: i32 = *s1;
        let s2 = c.borrow();
        let _x: i32 = *s1;
        let _y: i32 = *s2;
        let _x: i32 = *s1;
        let _y: i32 = *s2;
    }
    {
        let mut m = c.borrow_mut();
        let _z: i32 = *m;
        {
            let s: &i32 = &*m;
            let _x = *s;
        }
        *m = 23;
        let _z: i32 = *m;
    }
    {
        let s1 = c.borrow();
        let _x: i32 = *s1;
        let s2 = c.borrow();
        let _x: i32 = *s1;
        let _y: i32 = *s2;
        let _x: i32 = *s1;
        let _y: i32 = *s2;
    }
}

// Adding a Stacked Borrows protector for `Ref` would break this
fn ref_protector() {
    fn break_it(rc: &RefCell<i32>, r: Ref<'_, i32>) {
        // `r` has a shared reference, it is passed in as argument and hence
        // a protector is added that marks this memory as read-only for the entire
        // duration of this function.
        drop(r);
        // *oops* here we can mutate that memory.
        *rc.borrow_mut() = 2;
    }

    let rc = RefCell::new(0);
    break_it(&rc, rc.borrow())
}

fn ref_mut_protector() {
    fn break_it(rc: &RefCell<i32>, r: RefMut<'_, i32>) {
        // `r` has a shared reference, it is passed in as argument and hence
        // a protector is added that marks this memory as inaccessible for the entire
        // duration of this function
        drop(r);
        // *oops* here we can mutate that memory.
        *rc.borrow_mut() = 2;
    }

    let rc = RefCell::new(0);
    break_it(&rc, rc.borrow_mut())
}

/// Make sure we do not have bad enum layout optimizations.
fn rust_issue_68303() {
    let optional=Some(RefCell::new(false));
    let mut handle=optional.as_ref().unwrap().borrow_mut();
    assert!(optional.is_some());
    *handle=true;
}
