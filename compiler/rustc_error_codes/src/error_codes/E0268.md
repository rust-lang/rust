A loop keyword (`break` or `continue`) was used outside of a loop.

Erroneous code example:

```compile_fail,E0268
fn some_func() {
    break; // error: `break` outside of a loop
}
```

Without a loop to break out of or continue in, no sensible action can be taken.
Please verify that you are using `break` and `continue` only in loops. Example:

```
fn some_func() {
    for _ in 0..10 {
        break; // ok!
    }
}
```
