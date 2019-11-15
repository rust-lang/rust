The parameters of any trait method must match between a trait implementation
and the trait definition.

Erroneous code example:

```compile_fail,E0053
trait Foo {
    fn foo(x: u16);
    fn bar(&self);
}

struct Bar;

impl Foo for Bar {
    // error, expected u16, found i16
    fn foo(x: i16) { }

    // error, types differ in mutability
    fn bar(&mut self) { }
}
```
