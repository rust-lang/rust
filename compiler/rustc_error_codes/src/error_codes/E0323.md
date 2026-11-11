An associated const was implemented when another trait item was expected.

Erroneous code example:

```compile_fail,E0323
trait Foo {
    type N;
}

struct Bar;

impl Foo for Bar {
    const N : u32 = 0;
    // error: item `N` is an associated const, which doesn't match its
    //        trait `<Bar as Foo>`
}
```

Please verify that the associated const wasn't misspelled and the correct trait
was implemented. Example:

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
