//@ run-pass

use std::cell::RefCell;

struct S<'a>(i32, &'a RefCell<Vec<i32>>);

impl<'a> Drop for S<'a> {
    fn drop(&mut self) {
        self.1.borrow_mut().push(self.0);
    }
}

fn test(drops: &RefCell<Vec<i32>>) {
    let mut foo = None;
    let pfoo: *mut _ = &mut foo;

    match foo {
        None => (),
        _ => return,
    }

    // Both S(0) and S(1) should be dropped, but aren't.
    unsafe { *pfoo = Some((S(0, drops), S(1, drops))); }

    match foo {
        Some((_x, _)) => {}
        _ => {}
    }
}

fn main() {
    let drops = RefCell::new(Vec::new());
    test(&drops);

    // Ideally, we want this...
    //assert_eq!(*drops.borrow(), &[0, 1]);

    // But the delayed access through the raw pointer confuses drop elaboration,
    // causing S(1) to be leaked.
    assert_eq!(*drops.borrow(), &[0]);
}
