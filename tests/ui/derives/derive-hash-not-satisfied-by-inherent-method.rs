//! Regression test for https://github.com/rust-lang/rust/issues/21160
struct Bar;

impl Bar {
    fn hash<T>(&self, _: T) {}
}

#[derive(Hash)]
struct Foo(Bar);
//~^ error: `Bar: Hash` is not satisfied

fn main() {}
