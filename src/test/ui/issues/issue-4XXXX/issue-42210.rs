// run-pass
// Regression test for #42210.

// compile-flags: -g
// ignore-asmjs wasm2js does not support source maps yet

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
