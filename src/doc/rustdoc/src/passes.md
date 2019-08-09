# Passes

Rustdoc has a concept called "passes". These are transformations that
`rustdoc` runs on your documentation before producing its final output.

In addition to the passes below, check out the docs for these flags:

* [`--passes`](command-line-arguments.md#--passes-add-more-rustdoc-passes)
* [`--no-defaults`](command-line-arguments.md#--no-defaults-dont-run-default-passes)

## Default passes

By default, rustdoc will run some passes, namely:

* `strip-hidden`
* `strip-private`
* `collapse-docs`
* `unindent-comments`

However, `strip-private` implies `strip-private-imports`, and so effectively,
all passes are run by default.

## `strip-hidden`

This pass implements the `#[doc(hidden)]` attribute. When this pass runs, it
checks each item, and if it is annotated with this attribute, it removes it
from `rustdoc`'s output.

Without this pass, these items will remain in the output.

## `unindent-comments`

When you write a doc comment like this:

```rust,ignore
/// This is a documentation comment.
```

There's a space between the `///` and that `T`. That spacing isn't intended
to be a part of the output; it's there for humans, to help separate the doc
comment syntax from the text of the comment. This pass is what removes that
space.

The exact rules are left under-specified so that we can fix issues that we find.

Without this pass, the exact number of spaces is preserved.

## `collapse-docs`

With this pass, multiple `#[doc]` attributes are converted into one single
documentation string.

For example:

```rust,ignore
#[doc = "This is the first line."]
#[doc = "This is the second line."]
```

Gets collapsed into a single doc string of

```text
This is the first line.
This is the second line.
```

## `strip-private`

This removes documentation for any non-public items, so for example:

```rust,ignore
/// These are private docs.
struct Private;

/// These are public docs.
pub struct Public;
```

This pass removes the docs for `Private`, since they're not public.

This pass implies `strip-priv-imports`.

## `strip-priv-imports`

This is the same as `strip-private`, but for `extern crate` and `use`
statements instead of items.
