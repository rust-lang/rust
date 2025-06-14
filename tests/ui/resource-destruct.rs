//@ run-pass

#![allow(non_camel_case_types)]
use std::cell::Cell;

struct shrinky_pointer<'a> {
  i: &'a Cell<isize>,
}

impl<'a> Drop for shrinky_pointer<'a> {
    fn drop(&mut self) {
        println!("Hello!"); self.i.set(self.i.get() - 1);
    }
}

impl<'a> shrinky_pointer<'a> {
    pub fn look_at(&self) -> isize { return self.i.get(); }
}

fn shrinky_pointer(i: &Cell<isize>) -> shrinky_pointer<'_> {
    shrinky_pointer {
        i: i
    }
}

pub fn main() {
    let my_total = &Cell::new(10);
    { let pt = shrinky_pointer(my_total); assert_eq!(pt.look_at(), 10); }
    println!("my_total = {}", my_total.get());
    assert_eq!(my_total.get(), 9);
}
