A closure has been used as `static`.

Erroneous code example:

```compile_fail,E0697
fn main() {
    static || {}; // used as `static`
}
```

Closures cannot be used as `static`. They "save" the environment,
and as such a static closure would save only a static environment
which would consist only of variables with a static lifetime. Given
this it would be better to use a proper function. The easiest fix
is to remove the `static` keyword.
