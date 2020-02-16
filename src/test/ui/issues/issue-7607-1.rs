// FIXME: missing sysroot spans (#53081)
// ignore-i586-unknown-linux-gnu
// ignore-i586-unknown-linux-musl
// ignore-i686-unknown-linux-musl
struct Foo {
    x: isize
}

impl Fo { //~ ERROR cannot find type `Fo` in this scope
    fn foo() {}
}

fn main() {}
