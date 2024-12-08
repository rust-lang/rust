// Regression test for issue #72590
// Tests that we don't emit a spurious "size cannot be statically determined" error
//@ edition:2018

struct Foo {
    foo: Nonexistent, //~ ERROR cannot find
    other: str
}

struct Bar {
    test: Missing //~ ERROR cannot find
}

impl Foo {
    async fn frob(self) {} //~ ERROR the size
}

impl Bar {
    async fn myfn(self) {}
}

fn main() {}
