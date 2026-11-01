//@ run-pass

use std::cell::Cell;

struct R<'a> {
    b: &'a Cell<isize>,
}

impl<'a> Drop for R<'a> {
    fn drop(&mut self) {
        self.b.set(self.b.get() + 1);
    }
}

fn r(b: &Cell<isize>) -> R<'_> {
    R { b }
}

pub fn main() {
    let b = &Cell::new(0);
    {
        let _p = Some(r(b));
    }

    assert_eq!(b.get(), 1);
}
