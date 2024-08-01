# Advanced features

The features listed on this page fall outside the rest of the main categories.

## `#[cfg(doc)]`: Documenting platform-specific or feature-specific information

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

Please note that this `cfg` is not passed to doctests.

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

### Interactions between platform-specific docs

Rustdoc does not have a magic way to compile documentation 'as-if' you'd run it once for each
platform (such a magic wand has been called the ['holy grail of rustdoc'][#1998]). Instead,
it sees *all* of your code at once, the same way the Rust compiler would if you passed it
`--cfg doc`. The main difference is that rustdoc doesn't run all the compiler passes, meaning
that some invalid code won't emit an error.

[#1998]: https://github.com/rust-lang/rust/issues/1998

## Add aliases for an item in documentation search

This feature allows you to add alias(es) to an item when using the `rustdoc` search through the
`doc(alias)` attribute. Example:

```rust,no_run
#[doc(alias = "x")]
#[doc(alias = "big")]
pub struct BigX;
```

Then, when looking for it through the `rustdoc` search, if you enter "x" or
"big", search will show the `BigX` struct first.

There are some limitations on the doc alias names though: they cannot contain quotes (`'`, `"`)
or most whitespace. ASCII space is allowed if it does not start or end the alias.

You can add multiple aliases at the same time by using a list:

```rust,no_run
#[doc(alias("x", "big"))]
pub struct BigX;
```

## Custom search engines

If you find yourself often referencing online Rust docs you might enjoy using a custom search
engine. This allows you to use the navigation bar directly to search a `rustdoc` website.
Most browsers support this feature by letting you define a URL template containing `%s`
which will be substituted for the search term. As an example, for the standard library you could use
this template:

```text
https://doc.rust-lang.org/stable/std/?search=%s
```

Note that this will take you to a results page listing all matches. If you want to navigate to the first
result right away (which is often the best match) use the following instead:

```text
https://doc.rust-lang.org/stable/std/?search=%s&go_to_first=true
```

This URL adds the `go_to_first=true` query parameter which can be appended to any `rustdoc` search URL
to automatically go to the first result.

## `#[repr(transparent)]`: Documenting the transparent representation

You can read more about `#[repr(transparent)]` itself in the [Rust Reference][repr-trans-ref] and
in the [Rustonomicon][repr-trans-nomicon].

Since this representation is only considered part of the public ABI if the single field with non-trivial
size or alignment is public and if the documentation does not state otherwise, Rustdoc helpfully displays
the attribute if and only if the non-1-ZST field is public or at least one field is public in case all
fields are 1-ZST fields. The term *1-ZST* refers to types that are one-aligned and zero-sized.

It would seem that one can manually hide the attribute with `#[cfg_attr(not(doc), repr(transparent))]`
if one wishes to declare the representation as private even if the non-1-ZST field is public.
However, due to [current limitations][cross-crate-cfg-doc], this method is not always guaranteed to work.
Therefore, if you would like to do so, you should always write it down in prose independently of whether
you use `cfg_attr` or not.

[repr-trans-ref]: https://doc.rust-lang.org/reference/type-layout.html#the-transparent-representation
[repr-trans-nomicon]: https://doc.rust-lang.org/nomicon/other-reprs.html#reprtransparent
[cross-crate-cfg-doc]: https://github.com/rust-lang/rust/issues/114952
