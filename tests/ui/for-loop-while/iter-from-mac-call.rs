//! Tests for trait/type errors when dereferencing via macro in a for loop.

macro_rules! deref {
    ($e:expr) => {
        *$e
    };
}

fn f1(x: &mut i32) {
    for _item in deref!(x) {}
    //~^ ERROR `i32` is not an iterator
}

struct Wrapped(i32);

macro_rules! borrow_deref {
    ($e:expr) => {
        &mut *$e
    };
}

fn f2<'a>(mut iter: Box<dyn Iterator<Item = &'a mut i32>>) {
    for Wrapped(item) in borrow_deref!(iter) {
        //~^ ERROR mismatched types
        *item = 0
    }
}

fn main() {}
