//@ edition: 2018
//@ run-pass

#![feature(arbitrary_self_types)]

// tests that the referent type of a reference must be known to call methods on it

struct SmartPtr<T>(T);

impl<T> core::ops::Receiver for SmartPtr<T> {
    type Target = T;
}

impl<T> SmartPtr<T> {
    fn foo(&self) -> usize { 3 }
}

fn main() {
    let val = 1_u32;
    let ptr = SmartPtr(val);
    // Ensure calls to outer methods work even if inner methods can't be
    // resolved due to the type variable
    assert_eq!((ptr as SmartPtr<_>).foo(), 3);
}
