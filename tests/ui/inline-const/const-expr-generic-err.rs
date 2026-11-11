//@ build-fail

fn foo<T>() {
    const { assert!(std::mem::size_of::<T>() == 0); } //~ ERROR E0080
}

fn bar<const N: usize>() -> usize {
    const { N - 1 } //~ ERROR E0080
}

fn main() {
    foo::<i32>();
    bar::<0>();
}
