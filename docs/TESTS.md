# libsyntax2.0 testing infrastructure

Libsyntax2.0 tests are in the `tests/data` directory. Each test is a
pair of files, an `.rs` file with Rust code and a `.txt` file with a
human-readable representation of syntax tree.

The test suite is intended to be independent from a particular parser:
that's why it is just a list of files.

The test suite is intended to be progressive: that is, if you want to
write a Rust parser, you can TDD it by working through the test in
order. That's why each test file begins with the number. Generally,
tests should be added in order of the appearance of corresponding
functionality in libsytnax2.0. If a bug in parser is uncovered, a
**new** test should be created instead of modifying an existing one:
it is preferable to have a gazillion of small isolated test files,
rather than a single file which covers all edge cases. It's okay for
files to have the same name except for the leading number. In general,
test suite should be append-only: old tests should not be modified,
new tests should be created instead.

Note that only `ok` tests are normative: `err` tests test error
recovery and it is totally ok for a parser to not implement any error
recovery at all. However, for libsyntax2.0 we do care about error
recovery, and we do care about precise and useful error messages.

There are also so-called "inline tests". They appear as the comments
with a `test` header in the source code, like this:

```rust
// test fn_basic
// fn foo() {}
fn function(p: &mut Parser) {
    // ...
}
```

You can run `cargo collect-tests` command to collect all inline tests
into `tests/data/inline` directory. The main advantage of inline tests
is that they help to illustrate what the relevant code is doing.


Contribution opportunity: design and implement testing infrastructure
for validators.
