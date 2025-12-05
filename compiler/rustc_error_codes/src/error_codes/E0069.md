The compiler found a function whose body contains a `return;` statement but
whose return type is not `()`.

Erroneous code example:

```compile_fail,E0069
// error
fn foo() -> u8 {
    return;
}
```

Since `return;` is just like `return ();`, there is a mismatch between the
function's return type and the value being returned.
