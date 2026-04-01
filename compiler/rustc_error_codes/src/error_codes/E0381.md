It is not allowed to use or capture an uninitialized variable.

Erroneous code example:

```compile_fail,E0381
fn main() {
    let x: i32;
    let y = x; // error, use of possibly-uninitialized variable
}
```

To fix this, ensure that any declared variables are initialized before being
used. Example:

```
fn main() {
    let x: i32 = 0;
    let y = x; // ok!
}
```
