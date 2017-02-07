#![feature(plugin)]
#![plugin(clippy)]

#![deny(temporary_assignment)]

use std::ops::{Deref, DerefMut};

struct Struct {
    field: i32
}

struct Wrapper<'a> {
    inner: &'a mut Struct
}

impl<'a> Deref for Wrapper<'a> {
    type Target = Struct;
    fn deref(&self) -> &Struct { self.inner }
}

impl<'a> DerefMut for Wrapper<'a> {
    fn deref_mut(&mut self) -> &mut Struct { self.inner }
}

fn main() {
    let mut s = Struct { field: 0 };
    let mut t = (0, 0);

    Struct { field: 0 }.field = 1; //~ERROR assignment to temporary
    (0, 0).0 = 1; //~ERROR assignment to temporary

    // no error
    s.field = 1;
    t.0 = 1;
    Wrapper { inner: &mut s }.field = 1;
}
