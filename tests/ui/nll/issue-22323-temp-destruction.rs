// rust-lang/rust#22323: regression test demonstrating that NLL
// precisely tracks temporary destruction order.

//@ check-pass

fn main() {
    let _s = construct().borrow().consume_borrowed();
}

fn construct() -> Value { Value }

pub struct Value;

impl Value {
    fn borrow<'a>(&'a self) -> Borrowed<'a> { unimplemented!() }
}

pub struct Borrowed<'a> {
    _inner: Guard<'a, Value>,
}

impl<'a> Borrowed<'a> {
    fn consume_borrowed(self) -> String { unimplemented!() }
}

pub struct Guard<'a, T: ?Sized + 'a> {
    _lock: &'a T,
}

impl<'a, T: ?Sized> Drop for Guard<'a, T> { fn drop(&mut self) {} }
