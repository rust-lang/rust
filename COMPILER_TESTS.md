# Compiler Test Documentation

In the Rust project, we use a special set of commands embedded in
comments to test the Rust compiler. There are two groups of commands:

1. Header commands
2. Error info commands

Both types of commands are inside comments, but header commands should
be in a comment before any code.

## Summary of Error Info Commands

Error commands specify something about certain lines of the
program. They tell the test what kind of error and what message you
are expecting.

* `~`: Associates the following error level and message with the
  current line
* `~|`: Associates the following error level and message with the same
  line as the previous comment
* `~^`: Associates the following error level and message with the
  previous line. Each caret (`^`) that you add adds a line to this, so
  `~^^^^^^^` is seven lines up.

The error levels that you can have are:

1. `ERROR`
2. `WARNING`
3. `NOTE`
4. `HELP` and `SUGGESTION`*

\* **Note**: `SUGGESTION` must follow immediately after `HELP`.

## Summary of Header Commands

Header commands specify something about the entire test file as a
whole, instead of just a few lines inside the test.

* `ignore-X` where `X` is an architecture, OS or stage will ignore the test accordingly
* `ignore-pretty` will not compile the pretty-printed test (this is done to test the pretty-printer, but might not always work)
* `ignore-test` always ignores the test
* `ignore-lldb` and `ignore-gdb` will skip the debuginfo tests
* `min-{gdb,lldb}-version`
* `should-fail` indicates that the test should fail; used for "meta testing",
  where we test the compiletest program itself to check that it will generate
  errors in appropriate scenarios. This header is ignored for pretty-printer tests.
* `gate-test-X` where `X` is a feature marks the test as "gate test" for feature X.
  Such tests are supposed to ensure that the compiler errors when usage of a gated
  feature is attempted without the proper `#![feature(X)]` tag.
  Each unstable lang feature is required to have a gate test.

## Revisions

Certain classes of tests support "revisions" (as of the time of this
writing, this includes run-pass, compile-fail, run-fail, and
incremental, though incremental tests are somewhat
different). Revisions allow a single test file to be used for multiple
tests. This is done by adding a special header at the top of the file:

```
// revisions: foo bar baz
```

This will result in the test being compiled (and tested) three times,
once with `--cfg foo`, once with `--cfg bar`, and once with `--cfg
baz`. You can therefore use `#[cfg(foo)]` etc within the test to tweak
each of these results.

You can also customize headers and expected error messages to a particular
revision. To do this, add `[foo]` (or `bar`, `baz`, etc) after the `//`
comment, like so:

```
// A flag to pass in only for cfg `foo`:
//[foo]compile-flags: -Z verbose

#[cfg(foo)]
fn test_foo() {
    let x: usize = 32_u32; //[foo]~ ERROR mismatched types
}
```

Note that not all headers have meaning when customized to a revision.
For example, the `ignore-test` header (and all "ignore" headers)
currently only apply to the test as a whole, not to particular
revisions. The only headers that are intended to really work when
customized to a revision are error patterns and compiler flags.
