The `typeof` keyword is currently reserved but unimplemented.

Erroneous code example:

```compile_fail,E0516
fn main() {
    let x: typeof(92) = 92;
}
```

Try using type inference instead. Example:

```
fn main() {
    let x = 92;
}
```
