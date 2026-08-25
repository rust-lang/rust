#### Note: this error code is no longer emitted by the compiler.

The `pub` keyword was used inside a public enum.

Erroneous code example:

```compile_fail
pub enum Foo {
    pub Bar, // error: unnecessary `pub` visibility
}
```

Since the enum is already public, adding `pub` on one its elements is
unnecessary. Example:

```compile_fail
enum Foo {
    pub Bar, // not ok!
}
```

This is the correct syntax:

```
pub enum Foo {
    Bar, // ok!
}
```
