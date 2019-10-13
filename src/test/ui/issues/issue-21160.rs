// ignore-x86
// ^ due to stderr output differences
struct Bar;

impl Bar {
    fn hash<T>(&self, _: T) {}
}

#[derive(Hash)]
struct Foo(Bar);
//~^ error: `Bar: std::hash::Hash` is not satisfied

fn main() {}
