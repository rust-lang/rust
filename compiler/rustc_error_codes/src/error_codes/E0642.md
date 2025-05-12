Trait methods currently cannot take patterns as arguments.

Erroneous code example:

```compile_fail,E0642
trait Foo {
    fn foo((x, y): (i32, i32)); // error: patterns aren't allowed
                                //        in trait methods
}
```

You can instead use a single name for the argument:

```
trait Foo {
    fn foo(x_and_y: (i32, i32)); // ok!
}
```
