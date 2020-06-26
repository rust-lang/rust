More than one explicit lifetime bound was used on a trait object.

Example of erroneous code:

```compile_fail,E0226
trait Foo {}

type T<'a, 'b> = dyn Foo + 'a + 'b; // error: Trait object `arg` has two
                                    //        lifetime bound, 'a and 'b.
```

Here `T` is a trait object with two explicit lifetime bounds, 'a and 'b.

Only a single explicit lifetime bound is permitted on trait objects.
To fix this error, consider removing one of the lifetime bounds:

```
trait Foo {}

type T<'a> = dyn Foo + 'a;
```
