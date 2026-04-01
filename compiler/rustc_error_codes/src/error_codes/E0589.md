The value of `N` that was specified for `repr(align(N))` was not a power
of two, or was greater than 2^29.

Erroneous code example:

```compile_fail,E0589
#[repr(align(15))] // error: invalid `repr(align)` attribute: not a power of two
enum Foo {
    Bar(u64),
}
```
