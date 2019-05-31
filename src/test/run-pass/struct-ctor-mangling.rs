fn size_of_val<T>(_: &T) -> usize {
    std::mem::size_of::<T>()
}

struct Foo(i64);

// Test that the (symbol) mangling of `Foo` (the `struct` type) and that of
// `typeof Foo` (the function type of the `struct` constructor) don't collide.
fn main() {
    size_of_val(&Foo(0));
    size_of_val(&Foo);
}
