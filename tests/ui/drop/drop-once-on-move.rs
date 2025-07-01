//@ run-pass

#![allow(non_camel_case_types)]
use std::cell::Cell;

#[derive(Debug)]
struct r<'a> {
    i: &'a Cell<isize>,
}

impl<'a> Drop for r<'a> {
    fn drop(&mut self) {
        self.i.set(self.i.get() + 1);
    }
}

fn r(i: &Cell<isize>) -> r<'_> {
    r {
        i: i
    }
}

pub fn main() {
    let i = &Cell::new(0);
    // Even though these look like copies, they are guaranteed not to be
    {
        let a = r(i);
        let b = (a, 10);
        let (c, _d) = b;
        println!("{:?}", c);
    }
    assert_eq!(i.get(), 1);
}
