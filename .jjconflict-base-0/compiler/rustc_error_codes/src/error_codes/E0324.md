A method was implemented when another trait item was expected.

Erroneous code example:

```compile_fail,E0324
struct Bar;

trait Foo {
    const N : u32;

    fn M();
}

impl Foo for Bar {
    fn N() {}
    // error: item `N` is an associated method, which doesn't match its
    //        trait `<Bar as Foo>`
}
```

To fix this error, please verify that the method name wasn't misspelled and
verify that you are indeed implementing the correct trait items. Example:

```
struct Bar;

trait Foo {
    const N : u32;

    fn M();
}

impl Foo for Bar {
    const N : u32 = 0;

    fn M() {} // ok!
}
```
