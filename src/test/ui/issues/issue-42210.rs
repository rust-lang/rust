// run-pass
// Regression test for #42210.

// compile-flags: -g

trait Foo {
    fn foo() { }
}

struct Bar;

trait Baz {
}

impl Foo for (Bar, dyn Baz) { }


fn main() {
    <(Bar, dyn Baz) as Foo>::foo()
}
