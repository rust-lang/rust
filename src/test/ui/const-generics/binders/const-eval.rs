// check-pass
#![feature(const_generics)]
#![allow(incomplete_features)]

trait Baz<const N: usize> {}

fn test<T>() {
    // FIXME(const_generics): This should error.
    let _a: Box<dyn for<'a> Baz<{
        let _: &'a ();
        std::mem::size_of::<T>()
    }>>;
}

fn main() {
    test::<u32>();
}
