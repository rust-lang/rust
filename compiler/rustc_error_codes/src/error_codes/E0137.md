#### Note: this error code is no longer emitted by the compiler.

More than one function was declared with the `#[main]` attribute.

Erroneous code example:

```compile_fail
#![feature(main)]

#[main]
fn foo() {}

#[main]
fn f() {} // error: multiple functions with a `#[main]` attribute
```

This error indicates that the compiler found multiple functions with the
`#[main]` attribute. This is an error because there must be a unique entry
point into a Rust program. Example:

```compile_fail
#![feature(main)]

#[main]
fn f() {} // ok!
```
