//@ check-pass

fn foo<T>() -> usize {
    const { std::mem::size_of::<T>() }
}

fn bar<const N: usize>() -> usize {
    const { N + 1 }
}

fn main() {
    foo::<i32>();
    bar::<1>();
}
