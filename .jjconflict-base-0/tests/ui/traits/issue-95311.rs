//@ check-pass

// Test to check that pointee trait doesn't let region variables escape into the cache

#![feature(ptr_metadata)]

trait Bar: Sized + 'static {}

struct Foo<B: Bar> {
    marker: std::marker::PhantomData<B>,
}

impl<B: Bar> Foo<B> {
    fn foo<T: ?Sized>(value: &T) {
        std::ptr::metadata(value);
    }
}

fn main() {}
