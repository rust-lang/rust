macro_rules! deref {
    ($e:expr) => { *$e };
}

fn f1<'a>(mut iter: Box<dyn Iterator<Item=&'a mut u8>>) {
    for item in deref!(iter) { *item = 0 }
    //~^ ERROR `dyn Iterator<Item = &'a mut u8>` is not an iterator
}

fn f2(x: &mut i32) {
    for _item in deref!(x) {}
    //~^ ERROR `i32` is not an iterator
}

struct Wrapped(i32);

macro_rules! borrow_deref {
    ($e:expr) => { &mut *$e };
}

fn f3<'a>(mut iter: Box<dyn Iterator<Item=&'a mut i32>>) {
    for Wrapped(item) in borrow_deref!(iter) { *item = 0 }
    //~^ ERROR mismatched types
}

fn main() {}
