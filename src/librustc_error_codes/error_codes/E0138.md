More than one function was declared with the `#[start]` attribute.

Erroneous code example:

```compile_fail,E0138
#![feature(start)]

#[start]
fn foo(argc: isize, argv: *const *const u8) -> isize {}

#[start]
fn f(argc: isize, argv: *const *const u8) -> isize {}
// error: multiple 'start' functions
```

This error indicates that the compiler found multiple functions with the
`#[start]` attribute. This is an error because there must be a unique entry
point into a Rust program. Example:

```
#![feature(start)]

#[start]
fn foo(argc: isize, argv: *const *const u8) -> isize { 0 } // ok!
```
