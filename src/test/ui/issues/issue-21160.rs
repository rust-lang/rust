struct Bar;

impl Bar {
    fn hash<T>(&self, _: T) {}
}

#[derive(Hash)]
struct Foo(Bar);
//~^ error: `Bar: std::hash::Hash` is not satisfied

fn main() {}
