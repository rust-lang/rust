trait Foo {
    type X;
    fn method(&self) {}
}

#[derive(Clone)]
struct Bar<T: Foo> {
    x: T::X,
}

struct NotClone;

impl Foo for NotClone {
    type X = i8;
}

fn main() {
    Bar::<NotClone> { x: 1 }.clone(); //~ ERROR
}
