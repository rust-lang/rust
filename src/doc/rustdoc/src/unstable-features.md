# Unstable features

Rustdoc is under active development, and like the Rust compiler, some features are only available
on nightly releases. Some of these features are new and need some more testing before they're able to be
released to the world at large, and some of them are tied to features in the Rust compiler that are unstable. Several features here require a matching `#![feature(...)]` attribute to
enable, and thus are more fully documented in the [Unstable Book]. Those sections will link over
there as necessary.

[Unstable Book]: ../unstable-book/index.html

## Nightly-gated functionality

These features just require a nightly build to operate. Unlike the other features on this page,
these don't need to be "turned on" with a command-line flag or a `#![feature(...)]` attribute in
your crate. This can give them some subtle fallback modes when used on a stable release, so be
careful!

### Error numbers for `compile-fail` doctests

As detailed in [the chapter on documentation tests][doctest-attributes], you can add a
`compile_fail` attribute to a doctest to state that the test should fail to compile. However, on
nightly, you can optionally add an error number to state that a doctest should emit a specific error
number:

[doctest-attributes]: documentation-tests.html#attributes

``````markdown
```compile_fail,E0044
extern { fn some_func<T>(x: T); }
```
``````

This is used by the error index to ensure that the samples that correspond to a given error number
properly emit that error code. However, these error codes aren't guaranteed to be the only thing
that a piece of code emits from version to version, so this is unlikely to be stabilized in the
future.

Attempting to use these error numbers on stable will result in the code sample being interpreted as
plain text.

### Linking to items by type

As designed in [RFC 1946], Rustdoc can parse paths to items when you use them as links. To resolve
these type names, it uses the items currently in-scope, either by declaration or by `use` statement.
For modules, the "active scope" depends on whether the documentation is written outside the module
(as `///` comments on the `mod` statement) or inside the module (at `//!` comments inside the file
or block). For all other items, it uses the enclosing module's scope.

[RFC 1946]: https://github.com/rust-lang/rfcs/pull/1946

For example, in the following code:

```rust
/// Does the thing.
pub fn do_the_thing(_: SomeType) {
    println!("Let's do the thing!");
}

/// Token you use to [`do_the_thing`].
pub struct SomeType;
```

The link to ``[`do_the_thing`]`` in `SomeType`'s docs will properly link to the page for `fn
do_the_thing`. Note that here, rustdoc will insert the link target for you, but manually writing the
target out also works:

```rust
pub mod some_module {
    /// Token you use to do the thing.
    pub struct SomeStruct;
}

/// Does the thing. Requires one [`SomeStruct`] for the thing to work.
///
/// [`SomeStruct`]: some_module::SomeStruct
pub fn do_the_thing(_: some_module::SomeStruct) {
    println!("Let's do the thing!");
}
```

For more details, check out [the RFC][RFC 1946], and see [the tracking issue][43466] for more
information about what parts of the feature are available.

[43466]: https://github.com/rust-lang/rust/issues/43466

## Extensions to the `#[doc]` attribute

These features operate by extending the `#[doc]` attribute, and thus can be caught by the compiler
and enabled with a `#![feature(...)]` attribute in your crate.

### Documenting platform-/feature-specific information

Because of the way Rustdoc documents a crate, the documentation it creates is specific to the target
rustc compiles for. Anything that's specific to any other target is dropped via `#[cfg]` attribute
processing early in the compilation process. However, Rustdoc has a trick up its sleeve to handle
platform-specific code if it *does* receive it.

Because Rustdoc doesn't need to fully compile a crate to binary, it replaces function bodies with
`loop {}` to prevent having to process more than necessary. This means that any code within a
function that requires platform-specific pieces is ignored. Combined with a special attribute,
`#[doc(cfg(...))]`, you can tell Rustdoc exactly which platform something is supposed to run on,
ensuring that doctests are only run on the appropriate platforms.

The `#[doc(cfg(...))]` attribute has another effect: When Rustdoc renders documentation for that
item, it will be accompanied by a banner explaining that the item is only available on certain
platforms.

For Rustdoc to document an item, it needs to see it, regardless of what platform it's currently
running on. To aid this, Rustdoc sets the flag `#[cfg(rustdoc)]` when running on your crate.
Combining this with the target platform of a given item allows it to appear when building your crate
normally on that platform, as well as when building documentation anywhere.

For example, `#[cfg(any(windows, rustdoc))]` will preserve the item either on Windows or during the
documentation process. Then, adding a new attribute `#[doc(cfg(windows))]` will tell Rustdoc that
the item is supposed to be used on Windows. For example:

```rust
#![feature(doc_cfg)]

/// Token struct that can only be used on Windows.
#[cfg(any(windows, rustdoc))]
#[doc(cfg(windows))]
pub struct WindowsToken;

/// Token struct that can only be used on Unix.
#[cfg(any(unix, rustdoc))]
#[doc(cfg(unix))]
pub struct UnixToken;
```

In this sample, the tokens will only appear on their respective platforms, but they will both appear
in documentation.

`#[doc(cfg(...))]` was introduced to be used by the standard library and currently requires the
`#![feature(doc_cfg)]` feature gate. For more information, see [its chapter in the Unstable
Book][unstable-doc-cfg] and [its tracking issue][issue-doc-cfg].

[unstable-doc-cfg]: ../unstable-book/language-features/doc-cfg.html
[issue-doc-cfg]: https://github.com/rust-lang/rust/issues/43781

### Adding your trait to the "Important Traits" dialog

Rustdoc keeps a list of a few traits that are believed to be "fundamental" to a given type when
implemented on it. These traits are intended to be the primary interface for their types, and are
often the only thing available to be documented on their types. For this reason, Rustdoc will track
when a given type implements one of these traits and call special attention to it when a function
returns one of these types. This is the "Important Traits" dialog, visible as a circle-i button next
to the function, which, when clicked, shows the dialog.

In the standard library, the traits that qualify for inclusion are `Iterator`, `io::Read`, and
`io::Write`. However, rather than being implemented as a hard-coded list, these traits have a
special marker attribute on them: `#[doc(spotlight)]`. This means that you could apply this
attribute to your own trait to include it in the "Important Traits" dialog in documentation.

The `#[doc(spotlight)]` attribute currently requires the `#![feature(doc_spotlight)]` feature gate.
For more information, see [its chapter in the Unstable Book][unstable-spotlight] and [its tracking
issue][issue-spotlight].

[unstable-spotlight]: ../unstable-book/language-features/doc-spotlight.html
[issue-spotlight]: https://github.com/rust-lang/rust/issues/45040

### Exclude certain dependencies from documentation

The standard library uses several dependencies which, in turn, use several types and traits from the
standard library. In addition, there are several compiler-internal crates that are not considered to
be part of the official standard library, and thus would be a distraction to include in
documentation. It's not enough to exclude their crate documentation, since information about trait
implementations appears on the pages for both the type and the trait, which can be in different
crates!

To prevent internal types from being included in documentation, the standard library adds an
attribute to their `extern crate` declarations: `#[doc(masked)]`. This causes Rustdoc to "mask out"
types from these crates when building lists of trait implementations.

The `#[doc(masked)]` attribute is intended to be used internally, and requires the
`#![feature(doc_masked)]` feature gate.  For more information, see [its chapter in the Unstable
Book][unstable-masked] and [its tracking issue][issue-masked].

[unstable-masked]: ../unstable-book/language-features/doc-masked.html
[issue-masked]: https://github.com/rust-lang/rust/issues/44027

### Include external files as API documentation

As designed in [RFC 1990], Rustdoc can read an external file to use as a type's documentation. This
is useful if certain documentation is so long that it would break the flow of reading the source.
Instead of writing it all inline, writing `#[doc(include = "sometype.md")]` (where `sometype.md` is
a file adjacent to the `lib.rs` for the crate) will ask Rustdoc to instead read that file and use it
as if it were written inline.

[RFC 1990]: https://github.com/rust-lang/rfcs/pull/1990

`#[doc(include = "...")]` currently requires the `#![feature(external_doc)]` feature gate. For more
information, see [its chapter in the Unstable Book][unstable-include] and [its tracking
issue][issue-include].

[unstable-include]: ../unstable-book/language-features/external-doc.html
[issue-include]: https://github.com/rust-lang/rust/issues/44732

### Add aliases for an item in documentation search

This feature allows you to add alias(es) to an item when using the `rustdoc` search through the
`doc(alias)` attribute. Example:

```rust,no_run
#![feature(doc_alias)]

#[doc(alias = "x")]
#[doc(alias = "big")]
pub struct BigX;
```

Then, when looking for it through the `rustdoc` search, if you enter "x" or
"big", search will show the `BigX` struct first.

## Unstable command-line arguments

These features are enabled by passing a command-line flag to Rustdoc, but the flags in question are
themselves marked as unstable. To use any of these options, pass `-Z unstable-options` as well as
the flag in question to Rustdoc on the command-line. To do this from Cargo, you can either use the
`RUSTDOCFLAGS` environment variable or the `cargo rustdoc` command.

### `--markdown-before-content`: include rendered Markdown before the content

Using this flag looks like this:

```bash
$ rustdoc src/lib.rs -Z unstable-options --markdown-before-content extra.md
$ rustdoc README.md -Z unstable-options --markdown-before-content extra.md
```

Just like `--html-before-content`, this allows you to insert extra content inside the `<body>` tag
but before the other content `rustdoc` would normally produce in the rendered documentation.
However, instead of directly inserting the file verbatim, `rustdoc` will pass the files through a
Markdown renderer before inserting the result into the file.

### `--markdown-after-content`: include rendered Markdown after the content

Using this flag looks like this:

```bash
$ rustdoc src/lib.rs -Z unstable-options --markdown-after-content extra.md
$ rustdoc README.md -Z unstable-options --markdown-after-content extra.md
```

Just like `--html-after-content`, this allows you to insert extra content before the `</body>` tag
but after the other content `rustdoc` would normally produce in the rendered documentation.
However, instead of directly inserting the file verbatim, `rustdoc` will pass the files through a
Markdown renderer before inserting the result into the file.

### `--playground-url`: control the location of the playground

Using this flag looks like this:

```bash
$ rustdoc src/lib.rs -Z unstable-options --playground-url https://play.rust-lang.org/
```

When rendering a crate's docs, this flag gives the base URL of the Rust Playground, to use for
generating `Run` buttons. Unlike `--markdown-playground-url`, this argument works for standalone
Markdown files *and* Rust crates. This works the same way as adding `#![doc(html_playground_url =
"url")]` to your crate root, as mentioned in [the chapter about the `#[doc]`
attribute][doc-playground]. Please be aware that the official Rust Playground at
https://play.rust-lang.org does not have every crate available, so if your examples require your
crate, make sure the playground you provide has your crate available.

[doc-playground]: the-doc-attribute.html#html_playground_url

If both `--playground-url` and `--markdown-playground-url` are present when rendering a standalone
Markdown file, the URL given to `--markdown-playground-url` will take precedence. If both
`--playground-url` and `#![doc(html_playground_url = "url")]` are present when rendering crate docs,
the attribute will take precedence.

### `--crate-version`: control the crate version

Using this flag looks like this:

```bash
$ rustdoc src/lib.rs -Z unstable-options --crate-version 1.3.37
```

When `rustdoc` receives this flag, it will print an extra "Version (version)" into the sidebar of
the crate root's docs. You can use this flag to differentiate between different versions of your
library's documentation.

### `--linker`: control the linker used for documentation tests

Using this flag looks like this:

```bash
$ rustdoc --test src/lib.rs -Z unstable-options --linker foo
$ rustdoc --test README.md -Z unstable-options --linker foo
```

When `rustdoc` runs your documentation tests, it needs to compile and link the tests as executables
before running them. This flag can be used to change the linker used on these executables. It's
equivalent to passing `-C linker=foo` to `rustc`.

### `--sort-modules-by-appearance`: control how items on module pages are sorted

Using this flag looks like this:

```bash
$ rustdoc src/lib.rs -Z unstable-options --sort-modules-by-appearance
```

Ordinarily, when `rustdoc` prints items in module pages, it will sort them alphabetically (taking
some consideration for their stability, and names that end in a number). Giving this flag to
`rustdoc` will disable this sorting and instead make it print the items in the order they appear in
the source.

### `--themes`: provide additional themes

Using this flag looks like this:

```bash
$ rustdoc src/lib.rs -Z unstable-options --themes theme.css
```

Giving this flag to `rustdoc` will make it copy your theme into the generated crate docs and enable
it in the theme selector. Note that `rustdoc` will reject your theme file if it doesn't style
everything the "light" theme does. See `--theme-checker` below for details.

### `--theme-checker`: verify theme CSS for validity

Using this flag looks like this:

```bash
$ rustdoc -Z unstable-options --theme-checker theme.css
```

Before including your theme in crate docs, `rustdoc` will compare all the CSS rules it contains
against the "light" theme included by default. Using this flag will allow you to see which rules are
missing if `rustdoc` rejects your theme.

### `--resource-suffix`: modifying the name of CSS/JavaScript in crate docs

Using this flag looks like this:

```bash
$ rustdoc src/lib.rs -Z unstable-options --resource-suffix suf
```

When rendering docs, `rustdoc` creates several CSS and JavaScript files as part of the output. Since
all these files are linked from every page, changing where they are can be cumbersome if you need to
specially cache them. This flag will rename all these files in the output to include the suffix in
the filename. For example, `light.css` would become `light-suf.css` with the above command.

### `--display-warnings`: display warnings when documenting or running documentation tests

Using this flag looks like this:

```bash
$ rustdoc src/lib.rs -Z unstable-options --display-warnings
$ rustdoc --test src/lib.rs -Z unstable-options --display-warnings
```

The intent behind this flag is to allow the user to see warnings that occur within their library or
their documentation tests, which are usually suppressed. However, [due to a
bug][issue-display-warnings], this flag doesn't 100% work as intended. See the linked issue for
details.

[issue-display-warnings]: https://github.com/rust-lang/rust/issues/41574

### `--extern-html-root-url`: control how rustdoc links to non-local crates

Using this flag looks like this:

```bash
$ rustdoc src/lib.rs -Z unstable-options --extern-html-root-url some-crate=https://example.com/some-crate/1.0.1
```

Ordinarily, when rustdoc wants to link to a type from a different crate, it looks in two places:
docs that already exist in the output directory, or the `#![doc(doc_html_root)]` set in the other
crate. However, if you want to link to docs that exist in neither of those places, you can use these
flags to control that behavior. When the `--extern-html-root-url` flag is given with a name matching
one of your dependencies, rustdoc use that URL for those docs. Keep in mind that if those docs exist
in the output directory, those local docs will still override this flag.

### `-Z force-unstable-if-unmarked`

Using this flag looks like this:

```bash
$ rustdoc src/lib.rs -Z force-unstable-if-unmarked
```

This is an internal flag intended for the standard library and compiler that applies an
`#[unstable]` attribute to any dependent crate that doesn't have another stability attribute. This
allows `rustdoc` to be able to generate documentation for the compiler crates and the standard
library, as an equivalent command-line argument is provided to `rustc` when building those crates.

### `--index-page`: provide a top-level landing page for docs

This feature allows you to generate an index-page with a given markdown file. A good example of it
is the [rust documentation index](https://doc.rust-lang.org/index.html).

With this, you'll have a page which you can custom as much as you want at the top of your crates.

Using `index-page` option enables `enable-index-page` option as well.

### `--enable-index-page`: generate a default index page for docs

This feature allows the generation of a default index-page which lists the generated crates.

### `--static-root-path`: control how static files are loaded in HTML output

Using this flag looks like this:

```bash
$ rustdoc src/lib.rs -Z unstable-options --static-root-path '/cache/'
```

This flag controls how rustdoc links to its static files on HTML pages. If you're hosting a lot of
crates' docs generated by the same version of rustdoc, you can use this flag to cache rustdoc's CSS,
JavaScript, and font files in a single location, rather than duplicating it once per "doc root"
(grouping of crate docs generated into the same output directory, like with `cargo doc`). Per-crate
files like the search index will still load from the documentation root, but anything that gets
renamed with `--resource-suffix` will load from the given path.

### `--persist-doctests`: persist doctest executables after running

Using this flag looks like this:

```bash
$ rustdoc src/lib.rs --test -Z unstable-options --persist-doctests target/rustdoctest
```

This flag allows you to keep doctest executables around after they're compiled or run.
Usually, rustdoc will immediately discard a compiled doctest after it's been tested, but
with this option, you can keep those binaries around for farther testing.

### `--show-coverage`: calculate the percentage of items with documentation

Using this flag looks like this:

```bash
$ rustdoc src/lib.rs -Z unstable-options --show-coverage
```

If you want to determine how many items in your crate are documented, pass this flag to rustdoc.
When it receives this flag, it will count the public items in your crate that have documentation,
and print out the counts and a percentage instead of generating docs.

Some methodology notes about what rustdoc counts in this metric:

* Rustdoc will only count items from your crate (i.e. items re-exported from other crates don't
  count).
* Docs written directly onto inherent impl blocks are not counted, even though their doc comments
  are displayed, because the common pattern in Rust code is to write all inherent methods into the
  same impl block.
* Items in a trait implementation are not counted, as those impls will inherit any docs from the
  trait itself.
* By default, only public items are counted. To count private items as well, pass
  `--document-private-items` at the same time.

Public items that are not documented can be seen with the built-in `missing_docs` lint. Private
items that are not documented can be seen with Clippy's `missing_docs_in_private_items` lint.
