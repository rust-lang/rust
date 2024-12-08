Trait objects must include the `dyn` keyword.

Erroneous code example:

```edition2021,compile_fail,E0782
trait Foo {}
fn test(arg: Box<Foo>) {} // error!
```

Trait objects are a way to call methods on types that are not known until
runtime but conform to some trait.

Trait objects should be formed with `Box<dyn Foo>`, but in the code above
`dyn` is left off.

This makes it harder to see that `arg` is a trait object and not a
simply a heap allocated type called `Foo`.

To fix this issue, add `dyn` before the trait name.

```edition2021
trait Foo {}
fn test(arg: Box<dyn Foo>) {} // ok!
```

This used to be allowed before edition 2021, but is now an error.
