# The `#[doc]` attribute

The `#[doc]` attribute lets you control various aspects of how `rustdoc` does
its job.

The most basic function of `#[doc]` is to handle the actual documentation
text. That is, `///` is syntax sugar for `#[doc]` (as is `//!` for `#![doc]`).
This means that these two are the same:

```rust,no_run
/// This is a doc comment.
#[doc = r" This is a doc comment."]
# fn f() {}
```

(Note the leading space and the raw string literal in the attribute version.)

In most cases, `///` is easier to use than `#[doc]`. One case where the latter is easier is
when generating documentation in macros; the `collapse-docs` pass will combine multiple
`#[doc]` attributes into a single doc comment, letting you generate code like this:

```rust,no_run
#[doc = "This is"]
#[doc = " a "]
#[doc = "doc comment"]
# fn f() {}
```

Which can feel more flexible. Note that this would generate this:

```rust,no_run
#[doc = "This is\n a \ndoc comment"]
# fn f() {}
```

but given that docs are rendered via Markdown, it will remove these newlines.

Another use case is for including external files as documentation:

```rust,no_run
#[doc = include_str!("../../README.md")]
# fn f() {}
```

The `doc` attribute has more options though! These don't involve the text of
the output, but instead, various aspects of the presentation of the output.
We've split them into two kinds below: attributes that are useful at the
crate level, and ones that are useful at the item level.

## At the crate level

These options control how the docs look at a crate level.

### `html_favicon_url`

This form of the `doc` attribute lets you control the favicon of your docs.

```rust,no_run
#![doc(html_favicon_url = "https://example.com/favicon.ico")]
```

This will put `<link rel="icon" href="{}">` into your docs, where
the string for the attribute goes into the `{}`.

If you don't use this attribute, a default favicon will be used.

### `html_logo_url`

This form of the `doc` attribute lets you control the logo in the upper
left hand side of the docs.

```rust,no_run
#![doc(html_logo_url = "https://example.com/logo.jpg")]
```

This will put `<a href='../index.html'><img src='{}' alt='logo' width='100'></a>` into
your docs, where the string for the attribute goes into the `{}`.

If you don't use this attribute, there will be no logo.

### `html_playground_url`

This form of the `doc` attribute lets you control where the "run" buttons
on your documentation examples make requests to.

```rust,no_run
#![doc(html_playground_url = "https://playground.example.com/")]
```

Now, when you press "run", the button will make a request to this domain. The request
URL will contain 3 query parameters:
1. `code` for the code in the documentation
2. `version` for the Rust channel, e.g. nightly, which is decided by whether `code` contain unstable features
3. `edition` for the Rust edition, e.g. 2024

If you don't use this attribute, there will be no run buttons.

### `issue_tracker_base_url`

This form of the `doc` attribute is mostly only useful for the standard library;
When a feature is unstable, an issue number for tracking the feature must be
given. `rustdoc` uses this number, plus the base URL given here, to link to
the tracking issue.

```rust,no_run
#![doc(issue_tracker_base_url = "https://github.com/rust-lang/rust/issues/")]
```

### `html_root_url`

The `#[doc(html_root_url = "…")]` attribute value indicates the URL for
generating links to external crates. When rustdoc needs to generate a link to
an item in an external crate, it will first check if the extern crate has been
documented locally on-disk, and if so link directly to it. Failing that, it
will use the URL given by the `--extern-html-root-url` command-line flag if
available. If that is not available, then it will use the `html_root_url`
value in the extern crate if it is available. If that is not available, then
the extern items will not be linked.

```rust,no_run
#![doc(html_root_url = "https://docs.rs/serde/1.0")]
```

### `html_no_source`

By default, `rustdoc` will include the source code of your program, with links
to it in the docs. But if you include this:

```rust,no_run
#![doc(html_no_source)]
```

it will not.

### `test(no_crate_inject)`

By default, `rustdoc` will automatically add a line with `extern crate my_crate;` into each doctest.
But if you include this:

```rust,no_run
#![doc(test(no_crate_inject))]
```

it will not.

## At the item level

These forms of the `#[doc]` attribute are used on individual items, to control how
they are documented.

### `inline` and `no_inline`

<span id="docno_inlinedocinline"></span>

These attributes are used on `use` statements, and control where the documentation shows
up. For example, consider this Rust code:

```rust,no_run
pub use bar::Bar;

/// bar docs
pub mod bar {
    /// the docs for Bar
    pub struct Bar;
}
# fn main() {}
```

The documentation will generate a "Re-exports" section, and say `pub use bar::Bar;`, where
`Bar` is a link to its page.

If we change the `use` line like this:

```rust,no_run
#[doc(inline)]
pub use bar::Bar;
# pub mod bar { pub struct Bar; }
# fn main() {}
```

Instead, `Bar` will appear in a `Structs` section, just like `Bar` was defined at the
top level, rather than `pub use`'d.

Let's change our original example, by making `bar` private:

```rust,no_run
pub use bar::Bar;

/// bar docs
mod bar {
    /// the docs for Bar
    pub struct Bar;
}
# fn main() {}
```

Here, because `bar` is not public, `bar` wouldn't have its own page, so there's nowhere
to link to. `rustdoc` will inline these definitions, and so we end up in the same case
as the `#[doc(inline)]` above; `Bar` is in a `Structs` section, as if it were defined at
the top level. If we add the `no_inline` form of the attribute:

```rust,no_run
#[doc(no_inline)]
pub use bar::Bar;

/// bar docs
mod bar {
    /// the docs for Bar
    pub struct Bar;
}
# fn main() {}
```

Now we'll have a `Re-exports` line, and `Bar` will not link to anywhere.

One special case: In Rust 2018 and later, if you `pub use` one of your dependencies, `rustdoc` will
not eagerly inline it as a module unless you add `#[doc(inline)]`.

If you want to know more about inlining rules, take a look at the
[`re-exports` chapter](./re-exports.md).

### `hidden`

<span id="dochidden"></span>

Any item annotated with `#[doc(hidden)]` will not appear in the documentation,
unless the [`--document-hidden-items`](../unstable-features.md#document-hidden-items) flag is used.

You can find more information in the [`re-exports` chapter](./re-exports.md).

### `alias`

This attribute adds an alias in the search index.

Let's take an example:

```rust,no_run
#[doc(alias = "TheAlias")]
pub struct SomeType;
```

So now, if you enter "TheAlias" in the search, it'll display `SomeType`.
Of course, if you enter `SomeType` it'll return `SomeType` as expected!

#### FFI example

This doc attribute is especially useful when writing bindings for a C library.
For example, let's say we have a C function that looks like this:

```c
int lib_name_do_something(Obj *obj);
```

It takes a pointer to an `Obj` type and returns an integer. In Rust, it might
be written like this:

```ignore (using non-existing ffi types)
pub struct Obj {
    inner: *mut ffi::Obj,
}

impl Obj {
    pub fn do_something(&mut self) -> i32 {
        unsafe { ffi::lib_name_do_something(self.inner) }
    }
}
```

The function has been turned into a method to make it more convenient to use.
However, if you want to look for the Rust equivalent of `lib_name_do_something`,
you have no way to do so.

To get around this limitation, we just add `#[doc(alias = "lib_name_do_something")]`
on the `do_something` method and then it's all good!
Users can now look for `lib_name_do_something` in our crate directly and find
`Obj::do_something`.

### `test(attr(...))`

This form of the `doc` attribute allows you to add arbitrary attributes to all your doctests. For
example, if you want your doctests to fail if they have dead code, you could add this:

```rust,no_run
#![doc(test(attr(deny(dead_code))))]

mod my_mod {
    #![doc(test(attr(allow(dead_code))))] // but allow `dead_code` for this module
}
```

`test(attr(..))` attributes are appended to the parent module's, they do not replace the current
list of attributes. In the previous example, both attributes would be present:

```rust,no_run
// For every doctest in `my_mod`

#![deny(dead_code)] // from the crate-root
#![allow(dead_code)] // from `my_mod`
```

### `#[doc(cfg)]` and `#[doc(auto_cfg)]`

This feature aims at providing rustdoc users the possibility to add visual markers to the rendered documentation to know under which conditions an item is available.

It does not aim to allow having a same item with different `cfg`s to appear more than once in the generated documentation.

It does not aim to document items which are *inactive* under the current configuration (i.e., “`cfg`ed out”).

This features adds the following attributes:

 * `#[doc(auto_cfg)]`/`#[doc(auto_cfg = true)]`/`#[doc(auto_cfg = false)]`
 * `#[doc(cfg(...))]`
 * `#![doc(auto_cfg(hide(...)))]` / `#[doc(auto_cfg(show(...)))]`

All of these attributes can be added to a module or to the crate root, and they will be inherited by the child items unless another attribute overrides it. This is why "opposite" attributes like `auto_cfg(hide(...))` and `auto_cfg(show(...))` are provided: they allow a child item to override its parent.

#### `#[doc(cfg(...))]`

This attribute provides a standardized format to override `#[cfg()]` attributes to document conditionally available items. Example:

```rust,ignore (nightly)
// the "real" cfg condition
#[cfg(feature = "futures-io")]
// the `doc(cfg())` so it's displayed to the readers
#[doc(cfg(feature = "futures-io"))]
pub mod futures {}
```

It will display in the documentation for this module:

```text
This is supported on feature="futures-io" only.
```

You can use it to display information in generated documentation, whether or not there is a `#[cfg()]` attribute:

```rust,ignore (nightly)
#[doc(cfg(feature = "futures-io"))]
pub mod futures {}
```

It will be displayed exactly the same as the previous code.

This attribute has the same syntax as conditional compilation, but it only causes documentation to be added. This means `#[doc(cfg(not(windows)))]` will not cause your docs to be hidden on non-windows targets, even though `#[cfg(not(windows))]` does do that.

If `doc(auto_cfg)` is enabled on the item, `doc(cfg)` will override it anyway so in the two previous examples, even if the `doc(auto_cfg)` feature was enabled, it would still display the same thing.

This attribute works on modules and on items.

#### `#[doc(auto_cfg(hide(...)))]`

This attribute is used to prevent some `cfg` to be generated in the visual markers. It only applies to `#[doc(auto_cfg = true)]`, not to `#[doc(cfg(...))]`. So in the previous example:

```rust,ignore (nightly)
#[cfg(any(unix, feature = "futures-io"))]
pub mod futures {}
```

It currently displays both `unix` and `feature = "futures-io"` into the documentation, which is not great. To prevent the `unix` cfg to ever be displayed, you can use this attribute at the crate root level:

```rust,ignore (nightly)
#![doc(auto_cfg(hide(unix)))]
```

Or directly on a given item/module as it covers any of the item's descendants:

```rust,ignore (nightly)
#[doc(auto_cfg(hide(unix)))]
#[cfg(any(unix, feature = "futures-io"))]
pub mod futures {
    // `futures` and all its descendants won't display "unix" in their cfgs.
}
```

Then, the `unix` cfg will never be displayed into the documentation.

Rustdoc currently hides `doc` and `doctest` attributes by default and reserves the right to change the list of "hidden by default" attributes.

The attribute accepts only a list of identifiers or key/value items. So you can write:

```rust,ignore (nightly)
#[doc(auto_cfg(hide(unix, doctest, feature = "something")))]
#[doc(auto_cfg(hide()))]
```

But you cannot write:

```rust,ignore (nightly)
#[doc(auto_cfg(hide(not(unix))))]
```

So if we use `doc(auto_cfg(hide(unix)))`, it means it will hide all mentions of `unix`:

```rust,ignore (nightly)
#[cfg(unix)] // nothing displayed
#[cfg(any(unix))] // nothing displayed
#[cfg(any(unix, windows))] // only `windows` displayed
```

However, it only impacts the `unix` cfg, not the feature:

```rust,ignore (nightly)
#[cfg(feature = "unix")] // `feature = "unix"` is displayed
```

If `cfg_auto(show(...))` and `cfg_auto(hide(...))` are used to show/hide a same `cfg` on a same item, it'll emit an error. Example:

```rust,ignore (nightly)
#[doc(auto_cfg(hide(unix)))]
#[doc(auto_cfg(show(unix)))] // Error!
pub fn foo() {}
```

Using this attribute will re-enable `auto_cfg` if it was disabled at this location:

```rust,ignore (nightly)
#[doc(auto_cfg = false)] // Disabling `auto_cfg`
pub fn foo() {}
```

And using `doc(auto_cfg)` will re-enable it:

```rust,ignore (nightly)
#[doc(auto_cfg = false)] // Disabling `auto_cfg`
pub mod module {
    #[doc(auto_cfg(hide(unix)))] // `auto_cfg` is re-enabled.
    pub fn foo() {}
}
```

However, using `doc(auto_cfg = ...)` and `doc(auto_cfg(...))` on the same item will emit an error:

```rust,ignore (nightly)
#[doc(auto_cfg = false)]
#[doc(auto_cfg(hide(unix)))] // error
pub fn foo() {}
```

The reason behind this is that `doc(auto_cfg = ...)` enables or disables the feature, whereas `doc(auto_cfg(...))` enables it unconditionally, making the first attribute to appear useless as it will be overidden by the next `doc(auto_cfg)` attribute.

#### `#[doc(auto_cfg(show(...)))]`

This attribute does the opposite of `#[doc(auto_cfg(hide(...)))]`: if you used `#[doc(auto_cfg(hide(...)))]` and want to revert its effect on an item and its descendants, you can use `#[doc(auto_cfg(show(...)))]`.
It only applies to `#[doc(auto_cfg = true)]`, not to `#[doc(cfg(...))]`.

For example:

```rust,ignore (nightly)
#[doc(auto_cfg(hide(unix)))]
#[cfg(any(unix, feature = "futures-io"))]
pub mod futures {
    // `futures` and all its descendants won't display "unix" in their cfgs.
    #[doc(auto_cfg(show(unix)))]
    pub mod child {
        // `child` and all its descendants will display "unix" in their cfgs.
    }
}
```

The attribute accepts only a list of identifiers or key/value items. So you can write:

```rust,ignore (nightly)
#[doc(auto_cfg(show(unix, doctest, feature = "something")))]
#[doc(auto_cfg(show()))]
```

But you cannot write:

```rust,ignore (nightly)
#[doc(auto_cfg(show(not(unix))))]
```

If `auto_cfg(show(...))` and `auto_cfg(hide(...))` are used to show/hide a same `cfg` on a same item, it'll emit an error. Example:

```rust,ignore (nightly)
#[doc(auto_cfg(show(unix)))]
#[doc(auto_cfg(hide(unix)))] // Error!
pub fn foo() {}
```

Using this attribute will re-enable `auto_cfg` if it was disabled at this location:

```rust,ignore (nightly)
#[doc(auto_cfg = false)] // Disabling `auto_cfg`
#[doc(auto_cfg(show(unix)))] // `auto_cfg` is re-enabled.
pub fn foo() {}
```

#### `#[doc(auto_cfg)`/`#[doc(auto_cfg = true)]`/`#[doc(auto_cfg = false)]`

By default, `#[doc(auto_cfg)]` is enabled at the crate-level. When it's enabled, Rustdoc will automatically display `cfg(...)` compatibility information as-if the same `#[doc(cfg(...))]` had been specified.

This attribute impacts the item on which it is used and its descendants.

So if we take back the previous example:

```rust
#[cfg(feature = "futures-io")]
pub mod futures {}
```

There's no need to "duplicate" the `cfg` into a `doc(cfg())` to make Rustdoc display it.

In some situations, the detailed conditional compilation rules used to implement the feature might not serve as good documentation (for example, the list of supported platforms might be very long, and it might be better to document them in one place). To turn it off, add the `#[doc(auto_cfg = false)]` attribute on the item.

If no argument is specified (ie `#[doc(auto_cfg)]`), it's the same as writing `#[doc(auto_cfg = true)]`.

#### Inheritance

Rustdoc merges `cfg` attributes from parent modules to its children. For example, in this case, the module `non_unix` will describe the entire compatibility matrix for the module, and not just its directly attached information:

```rust,ignore (nightly)
#[doc(cfg(any(windows, unix)))]
pub mod desktop {
    #[doc(cfg(not(unix)))]
    pub mod non_unix {
        // ...
    }
}
```

This code will display:

```text
Available on (Windows or Unix) and non-Unix only.
```

#### Re-exports and inlining

`cfg` attributes of a re-export are never merged with the re-exported item(s) attributes except if the re-export has the `#[doc(inline)]` attribute. In this case, the `cfg` of the re-exported item will be merged with the re-export's.

When talking about "attributes merge", we mean that if the re-export has `#[cfg(unix)]` and the re-exported item has `#[cfg(feature = "foo")]`, you will only see `cfg(unix)` on the re-export and only `cfg(feature = "foo")` on the re-exported item, unless the re-export has `#[doc(inline)]`, then you will only see the re-exported item with both `cfg(unix)` and `cfg(feature = "foo")`.

Example:

```rust,ignore (nightly)
#[doc(cfg(any(windows, unix)))]
pub mod desktop {
    #[doc(cfg(not(unix)))]
    pub mod non_unix {
        // code
    }
}

#[doc(cfg(target_os = "freebsd"))]
pub use desktop::non_unix as non_unix_desktop;
#[doc(cfg(target_os = "macos"))]
#[doc(inline)]
pub use desktop::non_unix as inlined_non_unix_desktop;
```

In this example, `non_unix_desktop` will only display `cfg(target_os = "freeebsd")` and not display any `cfg` from `desktop::non_unix`.

On the contrary, `inlined_non_unix_desktop` will have cfgs from both the re-export and the re-exported item.

So that also means that if a crate re-exports a foreign item, unless it has `#[doc(inline)]`, the `cfg` and `doc(cfg)` attributes will not be visible:

```rust,ignore (nightly)
// dep:
#[cfg(feature = "a")]
pub struct S;

// crate using dep:

// There will be no mention of `feature = "a"` in the documentation.
pub use dep::S as Y;
```
