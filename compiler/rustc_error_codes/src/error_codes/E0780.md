Cannot use `doc(inline)` with anonymous imports

Erroneous code example:

```ignore (cannot-doctest-multicrate-project)

#[doc(inline)] // error: invalid doc argument
pub use foo::Foo as _;
```

Anonymous imports are always rendered with `#[doc(no_inline)]`. To fix this
error, remove the `#[doc(inline)]` attribute.

Example:

```ignore (cannot-doctest-multicrate-project)

pub use foo::Foo as _;
```
