# Macros

A number of minor features of Rust are not central enough to have their own
syntax, and yet are not implementable as functions. Instead, they are given
names, and invoked through a consistent syntax: `some_extension!(...)`.

Users of `rustc` can define new macros in two ways:

* [Macros] define new syntax in a higher-level,
  declarative way.
* [Procedural Macros] can be used to implement custom derive.

And one unstable way: [compiler plugins].

[Macros]: ../book/macros.html
[Procedural Macros]: ../book/procedural-macros.html
[compiler plugins]: ../book/compiler-plugins.html
