An implementation of a trait doesn't match the type constraint.

Erroneous code example:

```compile_fail,E0326
trait Foo {
    const BAR: bool;
}

struct Bar;

impl Foo for Bar {
    const BAR: u32 = 5; // error, expected bool, found u32
}
```

The types of any associated constants in a trait implementation must match the
types in the trait definition.
