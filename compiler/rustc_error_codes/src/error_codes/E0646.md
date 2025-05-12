It is not possible to define `main` with a where clause.

Erroneous code example:

```compile_fail,E0646
fn main() where i32: Copy { // error: main function is not allowed to have
                            // a where clause
}
```
