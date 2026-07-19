//! Check that a `-> impl Iterator<Item = Ty>` return type mismatch points at the
//! method call in the returned expression's chain where `Iterator::Item` diverged
//! from the signature's expectation, instead of only pointing at the signature.
//! Regression test for <https://github.com/rust-lang/rust/issues/106993>.

fn foo(items: &mut Vec<u8>) {
    items.sort();
}

fn bar() -> impl Iterator<Item = i32> {
    //~^ ERROR expected `foo` to return `i32`, but it returns `()`
    let mut x: Vec<Vec<u8>> = vec![
        vec![0, 2, 1],
        vec![5, 4, 3],
    ];
    x.iter_mut().map(foo)
}

fn baz() -> impl Iterator<Item = i32> {
    //~^ ERROR expected `foo` to return `i32`, but it returns `()`
    let mut x: Vec<Vec<u8>> = vec![
        vec![0, 2, 1],
        vec![5, 4, 3],
    ];
    let it = x.iter_mut().map(foo);
    it
}

fn chained() -> impl Iterator<Item = i32> {
    //~^ ERROR expected `IntoIter<u32>` to be an iterator that yields `i32`, but it yields `u32`
    let x = vec![0u32, 1, 2];
    x.into_iter().filter(|x| *x > 0).map(|x| x.checked_add(1)).flatten()
}

fn main() {
    bar();
    baz();
    chained();
}
