# Unstable features

Rustdoc is under active developement, and like the Rust compiler, some features are only available
on the nightly releases. Some of these are new and need some more testing before they're able to get
released to the world at large, and some of them are tied to features in the Rust compiler that are
themselves unstable. Several features here require a matching `#![feature(...)]` attribute to
enable, and thus are more fully documented in the [Unstable Book]. Those sections will link over
there as necessary.

[Unstable Book]: ../unstable-book/

## Error numbers for `compile-fail` doctests

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
that a piece of code emits from version to version, so this in unlikely to be stabilized in the
future.

Attempting to use these error numbers on stable will result in the code sample being interpreted as
plain text.

## Linking to items by type

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
