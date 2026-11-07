//@ check-pass
//@ compile-flags: -Znext-solver

fn test<T: Iterator>(x: T::Item) -> impl Sized {
    x
}

fn main() {}
