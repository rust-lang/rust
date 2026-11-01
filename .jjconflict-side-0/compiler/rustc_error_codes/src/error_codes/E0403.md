Some type parameters have the same name.

Erroneous code example:

```compile_fail,E0403
fn f<T, T>(s: T, u: T) {} // error: the name `T` is already used for a generic
                          //        parameter in this item's generic parameters
```

Please verify that none of the type parameters are misspelled, and rename any
clashing parameters. Example:

```
fn f<T, Y>(s: T, u: Y) {} // ok!
```

Type parameters in an associated item also cannot shadow parameters from the
containing item:

```compile_fail,E0403
trait Foo<T> {
    fn do_something(&self) -> T;
    fn do_something_else<T: Clone>(&self, bar: T);
}
```
