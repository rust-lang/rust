//! Test that the symbol mangling of Foo-the-constructor-function versus Foo-the-type do not collide

//@ run-pass

fn size_of_val<T>(_: &T) -> usize {
    std::mem::size_of::<T>()
}

struct Foo(#[allow(dead_code)] i64);

fn main() {
    size_of_val(&Foo(0));
    size_of_val(&Foo);
}
