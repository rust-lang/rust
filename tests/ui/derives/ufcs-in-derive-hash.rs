//! Regression test for <https://github.com/rust-lang/rust/issues/21160>.
//! Ensure we don't use existing hash method internally in hash derive.

struct Bar;

impl Bar {
    fn hash<T>(&self, _: T) {}
}

#[derive(Hash)]
struct Foo(Bar);
//~^ error: `Bar: Hash` is not satisfied

fn main() {}
