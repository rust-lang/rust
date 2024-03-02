struct Bar;

impl Bar {
    fn hash<T>(&self, _: T) {}
}

#[derive(Hash)]
struct Foo(Bar);
//~^ ERROR trait `Hash` is not implemented for `Bar`

fn main() {}
