The type placeholder `_` was used within a type on an item's signature.

Erroneous code example:

```compile_fail,E0121
fn foo() -> _ { 5 } // error

static BAR: _ = "test"; // error
```

In those cases, you need to provide the type explicitly:

```
fn foo() -> i32 { 5 } // ok!

static BAR: &str = "test"; // ok!
```

The type placeholder `_` can be used outside item's signature as follows:

```
let x = "a4a".split('4')
    .collect::<Vec<_>>(); // No need to precise the Vec's generic type.
```
