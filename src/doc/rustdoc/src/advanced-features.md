# Advanced Features

The features listed on this page fall outside the rest of the main categories.

## `#[cfg(doc)]`: Documenting platform-/feature-specific information

For conditional compilation, Rustdoc treats your crate the same way the compiler does. Only things
from the host target are available (or from the given `--target` if present), and everything else is
"filtered out" from the crate. This can cause problems if your crate is providing different things
on different targets and you want your documentation to reflect all the available items you
provide.

If you want to make sure an item is seen by Rustdoc regardless of what platform it's targeting,
you can apply `#[cfg(doc)]` to it. Rustdoc sets this whenever it's building documentation, so
anything that uses that flag will make it into documentation it generates. To apply this to an item
with other `#[cfg]` filters on it, you can write something like `#[cfg(any(windows, doc))]`.
This will preserve the item either when built normally on Windows, or when being documented
anywhere.

Please note that this feature is not passed to doctests.

Example:

```rust
/// Token struct that can only be used on Windows.
#[cfg(any(windows, doc))]
pub struct WindowsToken;
/// Token struct that can only be used on Unix.
#[cfg(any(unix, doc))]
pub struct UnixToken;
```

Here, the respective tokens can only be used by dependent crates on their respective platforms, but
they will both appear in documentation.
