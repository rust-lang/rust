//@ check-pass
use std::marker::PhantomData;

struct Value {
    values: &'static [&'static Value],
}

// This `static` recursively points to itself through a promoted (the slice).
static VALUE: Value = Value {
    values: &[&VALUE],
};

// If we just unfold this type going down the first variant of every enum, we'll never stop; we'll
// never even encounter the same type a second time.
struct S<T: 'static>(&'static S<(T, T)>, PhantomData<T>);
const C: &Result<S<()>, ()> = &Err(());

fn main() {}
