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
