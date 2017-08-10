# The `#[doc]` attribute

The `#[doc]` attribute lets you control various aspects of how `rustdoc` does
its job. 

The most basic job of `#[doc]` is to be the way that the text of the documentation
is handled. That is, `///` is syntax sugar for `#[doc]`. This means that these two
are the same:

```rust,ignore
/// This is a doc comment.
#[doc = "This is a doc comment."]
```

In most cases, `///` is easier to use than `#[doc]`. One case where the latter is easier is
when generating documentation in macros; the `collapse-docs` pass will combine multiple
`#[doc]` attributes into a single doc comment, letting you generate code like this:

```rust,ignore
#[doc = "This is"]
#[doc = " a "]
#[doc = "doc comment"]
```

Which can feel more flexible.

The `doc` attribute has more options though! These don't involve the text of
the output, but instead, various aspects of the presentation of the output.
We've split them into two kinds below: attributes that are useful at the
crate level, and ones that are useful at the item level.

## At the crate level

These options control how the docs look at a macro level.

### `html_favicon_url`

This form of the `doc` attribute lets you control the favicon of your docs.

```rust,ignore
#![doc(html_favicon_url = "https://foo.com/favicon.ico")]
```

This will put `<link rel="shortcut icon" href="{}">` into your docs, where
the string for the attribute goes into the `{}`.

### `html_logo_url`

This form of the `doc` attribute lets you control the logo in the upper
left hand side of the docs.

```rust,ignore
#![doc(html_logo_url = "https://foo.com/logo.jpg")]
```

This will put `<a href='index.html'><img src='{}' alt='logo' width='100'></a>` into
your docs, where the string for the attribute goes into the `{}`.

### `html_playground_url`

This form of the `doc` attribute lets you control where the "run" buttons
on your documentation examples make requests to.

```rust,ignore
#![doc(html_playground_url = "https://playground.foo.com/")]
```

Now, when you press "run", the button will make a request to this domain.

### `issue_tracker_base_url`

This form of the `doc` attribute is mostly only useful for the standard library;
When a feature is unstable, an issue number for tracking the feature must be
given. `rustdoc` uses this number, plus the base URL given here, to link to
the tracking issue.

```rust,ignore
#![doc(issue_tracker_base_url = "https://github.com/foo/foo/issues/")]
```

### `html_no_source`

By default, `rustdoc` will include the source code of your program, with links
to it in the docs. But if you include this:

```rust,ignore
#![doc(html_no_source)]
```

it will not.

## At the item level

These forms of the `#[doc]` attribute are used on individual items, to control how
they are documented.

## `#[doc(no_inline)]`

## `#[doc(hidden)]`

Any item annotated with `#[doc(hidden)]` will not appear in the documentation, unless
the `strip-hidden` pass is removed.

## `#[doc(primitive)]`

Since primitive types are defined in the compiler, there's no place to attach documentation
attributes. This attribute is used by the standard library to provide a way to generate
documentation for primitive types.