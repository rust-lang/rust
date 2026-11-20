An associated constant whose name does not match any of the associated constants
in the trait was used when implementing the trait.

Erroneous code example:

```compile_fail,E0438
trait Foo {}

impl Foo for i32 {
    const BAR: bool = true;
}
```

Trait implementations can only implement associated constants that are
members of the trait in question.

The solution to this problem is to remove the extraneous associated constant:

```
trait Foo {}

impl Foo for i32 {}
```
