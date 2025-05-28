# The `rustdoc-json` test suite

This page is specifically about the test suite named `rustdoc-json`, which tests rustdoc's [json output].
For other test suites used for testing rustdoc, see [Rustdoc tests](../rustdoc.md#tests).

Tests are run with compiletest, and have access to the usuall set of [directives](../tests/directives.md).
Frequenly used directives here are:

- [`//@ aux-build`][aux-build] to have dependencies.
- `//@ edition: 2021` (or some other edition).
- `//@ compile-flags: --document-hidden-items` to enable [document private items].

Each crate's json output is checked by 2 programs: [jsondoclint] and [jsondocck].

## jsondoclint

[jsondoclint] checks that all [`Id`]s exist in the `index` (or `paths`).
This makes sure their are no dangling [`Id`]s.

<!-- TODO: It does some more things too?
Also, talk about how it works
 -->

## jsondocck

<!-- TODO: shlex, jsonpath, values, variables -->

### Directives

- `//@ has <path>`:: Checks `<path>` exists, i.e. matches at least 1 value.
- `//@ !has <path>`:: Checks `<path>` doesn't exist, i.e. matches 0 values.
- `//@ has <path> <value>`: Check `<path>` exists, and 1 of the matches is equal to the given `<value>` 
- `//@ !has <path> <value>`: Checks `<path>` exists, but none of the matches equal the given `<value>`.
- `//@ is <path> <value>`: Check `<path>` matches exacly one value, and it's equal to the given `<value>`.
- `//@ is <path> <value> <value>...`: Check that `<path>` matches to exactly every given `<value>`. 
   Ordering doesn't matter here.
- `//@ !is <path> <value>`: Check `<path>` matches exactly one value, and that value is not equal to the given `<value>`.
- `//@ count <path> <number>` Check that `<path>` matches to `<number>` of values.



[json output]: https://doc.rust-lang.org/nightly/rustdoc/unstable-features.html#json-output
[jsondocck]: https://github.com/rust-lang/rust/tree/master/src/tools/jsondocck
[jsondoclint]: https://github.com/rust-lang/rust/tree/master/src/tools/jsondoclint
[aux-build]: ../tests/compiletest.md#building-auxiliary-crates
[`Id`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustdoc_json_types/struct.Id.html
[document private items]: https://doc.rust-lang.org/nightly/rustdoc/command-line-arguments.html#--document-private-items-show-items-that-are-not-public
