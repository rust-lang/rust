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
    match foo {
        None => (),
        _ => return,
    }

    *(&mut foo) = Some((S(0, drops), S(1, drops))); // Both S(0) and S(1) should be dropped

    match foo {
        Some((_x, _)) => {}
        _ => {}
    }
}

fn main() {
    let drops = RefCell::new(Vec::new());
    test(&drops);
    assert_eq!(*drops.borrow(), &[0, 1]);
}
