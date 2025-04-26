An associated type whose name does not match any of the associated types
in the trait was used when implementing the trait.

Erroneous code example:

```compile_fail,E0437
trait Foo {}

impl Foo for i32 {
    type Bar = bool;
}
```

Trait implementations can only implement associated types that are members of
the trait in question.

The solution to this problem is to remove the extraneous associated type:

```
trait Foo {}

impl Foo for i32 {}
```
