// known-bug: unknown
// compile-flags: -Ztrait-solver=next

fn test<T: Iterator>(x: T::Item) -> impl Sized {
    x
}

fn main() {}
