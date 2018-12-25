// run-pass

#![allow(non_camel_case_types)]

use std::cell::Cell;

// Make sure that destructors get run on slice literals
struct foo<'a> {
    x: &'a Cell<isize>,
}

impl<'a> Drop for foo<'a> {
    fn drop(&mut self) {
        self.x.set(self.x.get() + 1);
    }
}

fn foo(x: &Cell<isize>) -> foo {
    foo {
        x: x
    }
}

pub fn main() {
    let x = &Cell::new(0);
    {
        let l = &[foo(x)];
        assert_eq!(l[0].x.get(), 0);
    }
    assert_eq!(x.get(), 1);
}
