//@ run-rustfix
#![allow(dead_code, unused_assignments, unused_variables)]

use std::ops::{Deref, DerefMut};

fn main() {
    let mut db = DataBox::new(Complex {
        b: true,
        s: Some(Simple { a: 0, b: false }),
    });

    if let Some(mut s) = db.s {
        //~^ ERROR cannot move out of dereference of `DataBox<Complex>`
        s.a = 1;
    }

    #[allow(unused_mut)]
    let mut clone = db.clone();
    if let Some(mut s) = clone.s {
        s.a = 1;
    }
    *db = clone; //~ ERROR use of partially moved value: `clone`
}

#[derive(Clone)]
struct Simple {
    a: u8,
    b: bool,
}

#[derive(Clone)]
struct Complex {
    b: bool,
    s: Option<Simple>,
}

struct DataBox<T> {
    d: T,
}

impl<T> DataBox<T> {
    fn new(d: T) -> DataBox<T> {
        DataBox { d }
    }
}

impl<T> Deref for DataBox<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.d
    }
}

impl<T> DerefMut for DataBox<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.d
    }
}
