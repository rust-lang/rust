// check-pass
// compile-flags: -Z trait-solver=chalk

struct Foo<'a, T> where Box<T>: Clone {
    _x: std::marker::PhantomData<&'a T>,
}

fn main() { }
