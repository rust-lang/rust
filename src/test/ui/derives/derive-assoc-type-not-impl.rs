// FIXME: missing sysroot spans (#53081)
// ignore-i586-unknown-linux-gnu
// ignore-i586-unknown-linux-musl
// ignore-i686-unknown-linux-musl

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
