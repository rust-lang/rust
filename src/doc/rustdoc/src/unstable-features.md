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
