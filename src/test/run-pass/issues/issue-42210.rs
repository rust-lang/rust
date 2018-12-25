// run-pass
// Regression test for #42210.

// compile-flags: -g

trait Foo {
    fn foo() { }
}

struct Bar;

trait Baz {
}

impl Foo for (Bar, Baz) { }


fn main() {
    <(Bar, Baz) as Foo>::foo()
}
