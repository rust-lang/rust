# The `rustdoc-json` test suite

This page is specifically about the test suite named `rustdoc-json`, which tests rustdoc's [json output].
For other test suites used for testing rustdoc, see [§Rustdoc test suites](../tests/compiletest.md#rustdoc-test-suites).

Tests are run with compiletest, and have access to the usual set of [directives](../tests/directives.md).
Frequenly used directives here are:

- [`//@ aux-build`][aux-build] to have dependencies.
- `//@ edition: 2021` (or some other edition).
- `//@ compile-flags: --document-hidden-items` to enable [document private items].

Each crate's json output is checked by 2 programs: [jsondoclint](#jsondocck) and [jsondocck](#jsondocck).

## jsondoclint

[jsondoclint] checks that all [`Id`]s exist in the `index` (or `paths`).
This makes sure there are no dangling [`Id`]s.

<!-- TODO: It does some more things too?
Also, talk about how it works
 -->

## jsondocck

[jsondocck] processes direcives given in comments, to assert that the values in the output are expected.
It's alot like [htmldocck](./rustdoc-test-suite.md) in that way.

It uses [JSONPath] as a query language, which takes a path, and returns a *list* of values that that path is said to match to.

### Directives

- `//@ has <path>`: Checks `<path>` exists, i.e. matches at least 1 value.
- `//@ !has <path>`: Checks `<path>` doesn't exist, i.e. matches 0 values.
- `//@ has <path> <value>`: Check `<path>` exists, and at least 1 of the matches is equal to the given `<value>` 
- `//@ !has <path> <value>`: Checks `<path>` exists, but none of the matches equal the given `<value>`.
- `//@ is <path> <value>`: Check `<path>` matches exactly one value, and it's equal to the given `<value>`.
- `//@ is <path> <value> <value>...`: Check that `<path>` matches to exactly every given `<value>`. 
   Ordering doesn't matter here.
- `//@ !is <path> <value>`: Check `<path>` matches exactly one value, and that value is not equal to the given `<value>`.
- `//@ count <path> <number>`: Check that `<path>` matches to `<number>` of values.
- `//@ set <name> = <path>`: Check that `<path>` matches exactly one value, and store that value to the variable called `<name>`.

These are defined in [`directive.rs`].

### Values

Values can be either JSON values, or variables.

- JSON values are JSON literals, e.g. `true`, `"string"`, `{"key": "value"}`. 
  These often need to be quoted using `'`, to be processed as 1 value. See [§Argument spliting](#argument-spliting)
- Variables can be used to store the value in one path, and use it in later queries.
  They are set with the `//@ set <name> = <path>` directive, and accessed with `$<name>`

  ```rust
  //@ set foo = $some.path
  //@ is $.some.other.path $foo
  ```

### Argument spliting

Arguments to directives are split using the [shlex] crate, which implements POSIX shell escaping.
This is because both `<path>` and `<value>` arguments to [directives](#directives) frequently have both
whitespace and quote marks.

To use the `@ is` with a `<path>` of `$.index[?(@.docs == "foo")].some.field` and a value of `"bar"` [^why_quote], you'd write:

```rust
//@ is '$.is[?(@.docs == "foo")].some.field' '"bar"'
```

[^why_quote]: The value needs to be `"bar"` *after* shlex splitting, because we
    it needs to be a JSON string value.

[json output]: https://doc.rust-lang.org/nightly/rustdoc/unstable-features.html#json-output
[jsondocck]: https://github.com/rust-lang/rust/tree/master/src/tools/jsondocck
[jsondoclint]: https://github.com/rust-lang/rust/tree/master/src/tools/jsondoclint
[aux-build]: ../tests/compiletest.md#building-auxiliary-crates
[`Id`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustdoc_json_types/struct.Id.html
[document private items]: https://doc.rust-lang.org/nightly/rustdoc/command-line-arguments.html#--document-private-items-show-items-that-are-not-public
[`directive.rs`]: https://github.com/rust-lang/rust/blob/master/src/tools/jsondocck/src/directive.rs
[shlex]: https://docs.rs/shlex/1.3.0/shlex/index.html
[JSONPath]: https://www.rfc-editor.org/rfc/rfc9535.html
