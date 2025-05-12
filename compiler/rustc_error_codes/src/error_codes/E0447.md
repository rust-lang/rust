#### Note: this error code is no longer emitted by the compiler.

The `pub` keyword was used inside a function.

Erroneous code example:

```
fn foo() {
    pub struct Bar; // error: visibility has no effect inside functions
}
```

Since we cannot access items defined inside a function, the visibility of its
items does not impact outer code. So using the `pub` keyword in this context
is invalid.
