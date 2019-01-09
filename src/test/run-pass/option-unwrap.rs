#![allow(non_camel_case_types)]
use std::cell::Cell;

struct dtor<'a> {
    x: &'a Cell<isize>,
}

impl<'a> Drop for dtor<'a> {
    fn drop(&mut self) {
        self.x.set(self.x.get() - 1);
    }
}

fn unwrap<T>(o: Option<T>) -> T {
    match o {
      Some(v) => v,
      None => panic!()
    }
}

pub fn main() {
    let x = &Cell::new(1);

    {
        let b = Some(dtor { x:x });
        let _c = unwrap(b);
    }

    assert_eq!(x.get(), 0);
}
