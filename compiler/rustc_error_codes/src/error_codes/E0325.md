An associated type was implemented when another trait item was expected.

Erroneous code example:

```compile_fail,E0325
struct Bar;

trait Foo {
    const N : u32;
}

impl Foo for Bar {
    type N = u32;
    // error: item `N` is an associated type, which doesn't match its
    //        trait `<Bar as Foo>`
}
```

Please verify that the associated type name wasn't misspelled and your
implementation corresponds to the trait definition. Example:

```
struct Bar;

trait Foo {
    type N;
}

impl Foo for Bar {
    type N = u32; // ok!
}
```

Or:

```
struct Bar;

trait Foo {
    const N : u32;
}

impl Foo for Bar {
    const N : u32 = 0; // ok!
}
```
