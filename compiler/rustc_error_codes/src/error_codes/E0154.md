#### Note: this error code is no longer emitted by the compiler.

Imports (`use` statements) are not allowed after non-item statements, such as
variable declarations and expression statements.

Here is an example that demonstrates the error:

```
fn f() {
    // Variable declaration before import
    let x = 0;
    use std::io::Read;
    // ...
}
```

The solution is to declare the imports at the top of the block, function, or
file.

Here is the previous example again, with the correct order:

```
fn f() {
    use std::io::Read;
    let x = 0;
    // ...
}
```

See the [Declaration Statements][declaration-statements] section of the
reference for more information about what constitutes an item declaration
and what does not.

[declaration-statements]: https://doc.rust-lang.org/reference/statements.html#declaration-statements
