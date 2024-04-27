//@ run-pass
#![allow(non_camel_case_types)]

// Ensures that class dtors run if the object is inside an enum
// variant

use std::cell::Cell;

type closable<'a> = &'a Cell<bool>;

struct close_res<'a> {
  i: closable<'a>,

}

impl<'a> Drop for close_res<'a> {
    fn drop(&mut self) {
        self.i.set(false);
    }
}

fn close_res(i: closable) -> close_res {
    close_res {
        i: i
    }
}

enum option<T> { none, some(#[allow(dead_code)] T), }

fn sink(_res: option<close_res>) { }

pub fn main() {
    let c = &Cell::new(true);
    sink(option::none);
    sink(option::some(close_res(c)));
    assert!(!c.get());
}
