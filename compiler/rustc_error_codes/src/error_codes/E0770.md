The type of a const parameter references other generic parameters.

Erroneous code example:

```compile_fail,E0770
fn foo<T, const N: T>() {} // error!
```

To fix this error, use a concrete type for the const parameter:

```
fn foo<T, const N: usize>() {}
```
