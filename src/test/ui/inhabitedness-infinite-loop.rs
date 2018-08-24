// error-pattern:reached recursion limit

#![feature(never_type)]
#![feature(exhaustive_patterns)]

struct Foo<'a, T: 'a> {
    ph: std::marker::PhantomData<T>,
    foo: &'a Foo<'a, (T, T)>,
}

fn wub(f: Foo<!>) {
    match f {}
}

fn main() {}

