#![allow(clippy::needless_if)]

#[derive(Default)]
struct A<T> {
    a: Vec<A<T>>,
    b: T,
}

fn main() {
    if let Ok(_) = Ok::<_, ()>(A::<String>::default()) {}
    //~^ ERROR: redundant pattern matching, consider using `is_ok()`
    //~| NOTE: `-D clippy::redundant-pattern-matching` implied by `-D warnings`
}
