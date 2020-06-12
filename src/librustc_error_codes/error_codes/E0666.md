`impl Trait` types cannot appear nested in the generic arguments of other
`impl Trait` types.

Erroneous code example:

```compile_fail,E0666
trait MyGenericTrait<T> {}
trait MyInnerTrait {}

fn foo(
    bar: impl MyGenericTrait<impl MyInnerTrait>, // error!
) {}
```

Type parameters for `impl Trait` types must be explicitly defined as named
generic parameters:

```
trait MyGenericTrait<T> {}
trait MyInnerTrait {}

fn foo<T: MyInnerTrait>(
    bar: impl MyGenericTrait<T>, // ok!
) {}
```
