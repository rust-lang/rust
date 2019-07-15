# `param_attrs`

The tracking issue for this feature is: [#60406]

[#60406]: https://github.com/rust-lang/rust/issues/60406

Allow attributes in formal function parameter position so external tools and compiler internals can
take advantage of the additional information that the parameters provide.

Enables finer conditional compilation with `#[cfg(..)]` and linting control of variables. Moreover,
opens the path to richer DSLs created by users.

------------------------

Example:

```rust
#![feature(param_attrs)]

fn len(
  #[cfg(windows)] slice: &[u16],
  #[cfg(not(windows))] slice: &[u8],
) -> usize
{
  slice.len()
}
```
