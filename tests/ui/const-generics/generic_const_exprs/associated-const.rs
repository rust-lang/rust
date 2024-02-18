//@ check-pass
struct Foo<T>(T);
impl<T> Foo<T> {
    const VALUE: usize = std::mem::size_of::<T>();
}

fn test<T>() {
    let _ = [0; Foo::<u8>::VALUE];
}

fn main() {}
