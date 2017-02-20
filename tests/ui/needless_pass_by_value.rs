#![feature(plugin)]
#![plugin(clippy)]

#![deny(needless_pass_by_value)]
#![allow(dead_code)]

// `v` will be warned
// `w`, `x` and `y` are allowed (moved or mutated)
fn foo<T: Default>(v: Vec<T>, w: Vec<T>, mut x: Vec<T>, y: Vec<T>) -> Vec<T> {
    assert_eq!(v.len(), 42);

    consume(w);

    x.push(T::default());

    y
}

fn consume<T>(_: T) {}

struct Wrapper(String);

fn bar(x: String, y: Wrapper) {
    assert_eq!(x.len(), 42);
    assert_eq!(y.0.len(), 42);
}

fn test_borrow_trait<T: std::borrow::Borrow<str>, U>(t: T, u: U) {
    // U implements `Borrow<U>`, but warned correctly
    println!("{}", t.borrow());
    consume(&u);
}

// ok
fn test_fn<F: Fn(i32) -> i32>(f: F) {
    f(1);
}

fn main() {}
