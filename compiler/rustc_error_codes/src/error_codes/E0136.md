#### Note: this error code is no longer emitted by the compiler.

More than one `main` function was found.

Erroneous code example:

```compile_fail
fn main() {
    // ...
}

// ...

fn main() { // error!
    // ...
}
```

A binary can only have one entry point, and by default that entry point is the
`main()` function. If there are multiple instances of this function, please
rename one of them.
