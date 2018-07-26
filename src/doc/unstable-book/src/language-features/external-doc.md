# `external_doc`

The tracking issue for this feature is: [#44732]

The `external_doc` feature allows the use of the `include` parameter to the `#[doc]` attribute, to
include external files in documentation. Use the attribute in place of, or in addition to, regular
doc comments and `#[doc]` attributes, and `rustdoc` will load the given file when it renders
documentation for your crate.

With the following files in the same directory:

`external-doc.md`:

```markdown
# My Awesome Type

This is the documentation for this spectacular type.
```

`lib.rs`:

```no_run (needs-external-files)
#![feature(external_doc)]

#[doc(include = "external-doc.md")]
pub struct MyAwesomeType;
```

`rustdoc` will load the file `external-doc.md` and use it as the documentation for the `MyAwesomeType`
struct.

When locating files, `rustdoc` will base paths in the `src/` directory, as if they were alongside the
`lib.rs` for your crate. So if you want a `docs/` folder to live alongside the `src/` directory,
start your paths with `../docs/` for `rustdoc` to properly find the file.

This feature was proposed in [RFC #1990] and initially implemented in PR [#44781].

[#44732]: https://github.com/rust-lang/rust/issues/44732
[RFC #1990]: https://github.com/rust-lang/rfcs/pull/1990
[#44781]: https://github.com/rust-lang/rust/pull/44781
