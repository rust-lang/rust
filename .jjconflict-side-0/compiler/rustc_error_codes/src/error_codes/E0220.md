The associated type used was not defined in the trait.

Erroneous code example:

```compile_fail,E0220
trait T1 {
    type Bar;
}

type Foo = T1<F=i32>; // error: associated type `F` not found for `T1`

// or:

trait T2 {
    type Bar;

    // error: Baz is used but not declared
    fn return_bool(&self, _: &Self::Bar, _: &Self::Baz) -> bool;
}
```

Make sure that you have defined the associated type in the trait body.
Also, verify that you used the right trait or you didn't misspell the
associated type name. Example:

```
trait T1 {
    type Bar;
}

type Foo = T1<Bar=i32>; // ok!

// or:

trait T2 {
    type Bar;
    type Baz; // we declare `Baz` in our trait.

    // and now we can use it here:
    fn return_bool(&self, _: &Self::Bar, _: &Self::Baz) -> bool;
}
```
