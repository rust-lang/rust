Items are missing in a trait implementation.

Erroneous code example:

```compile_fail,E0046
trait Foo {
    fn foo();
}

struct Bar;

impl Foo for Bar {}
// error: not all trait items implemented, missing: `foo`
```

When trying to make some type implement a trait `Foo`, you must, at minimum,
provide implementations for all of `Foo`'s required methods (meaning the
methods that do not have default implementations), as well as any required
trait items like associated types or constants. Example:

```
trait Foo {
    fn foo();
}

struct Bar;

impl Foo for Bar {
    fn foo() {} // ok!
}
```
