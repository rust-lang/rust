//@ compile-flags:-Zpolymorphize=on
//@ build-pass
//@ dont-check-compiler-stderr

fn test<T>() {
    std::mem::size_of::<T>();
}

pub fn foo<T>(_: T) -> &'static fn() {
    &(test::<T> as fn())
}

fn outer<T>() {
    foo(|| ());
}

fn main() {
    outer::<u8>();
}
