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
whole. They are normally put right after the copyright comment, e.g.:

```Rust
// Copyright blah blah blah
// except according to those terms.

// ignore-test This doesn't actually work
```

### Ignoring tests

These are used to ignore the test in some situations, which means the test won't
be compiled or run.

* `ignore-X` where `X` is a target detail or stage will ignore the test accordingly (see below)
* `ignore-pretty` will not compile the pretty-printed test (this is done to test the pretty-printer, but might not always work)
* `ignore-test` always ignores the test
* `ignore-lldb` and `ignore-gdb` will skip a debuginfo test on that debugger.

`only-X` is the opposite. The test will run only when `X` matches.

Some examples of `X` in `ignore-X`:

* Architecture: `aarch64`, `arm`, `asmjs`, `mips`, `wasm32`, `x86_64`, `x86`, ...
* OS: `android`, `emscripten`, `freebsd`, `ios`, `linux`, `macos`, `windows`, ...
* Environment (fourth word of the target triple): `gnu`, `msvc`, `musl`.
* Pointer width: `32bit`, `64bit`.
* Stage: `stage0`, `stage1`, `stage2`.

### Other Header Commands

* `min-{gdb,lldb}-version`
* `min-llvm-version`
* `compile-pass` for UI tests, indicates that the test is supposed
  to compile, as opposed to the default where the test is supposed to error out.
* `compile-flags` passes extra command-line args to the compiler,
  e.g. `compile-flags -g` which forces debuginfo to be enabled.
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

## Guide to the UI Tests

The UI tests are intended to capture the compiler's complete output,
so that we can test all aspects of the presentation. They work by
compiling a file (e.g., `ui/hello_world/main.rs`), capturing the output,
and then applying some normalization (see below). This normalized
result is then compared against reference files named
`ui/hello_world/main.stderr` and `ui/hello_world/main.stdout`. If either of
those files doesn't exist, the output must be empty. If the test run
fails, we will print out the current output, but it is also saved in
`build/<target-triple>/test/ui/hello_world/main.stdout` (this path is
printed as part of the test failure message), so you can run `diff` and
so forth.

Normally, the test-runner checks that UI tests fail compilation. If you want
to do a UI test for code that *compiles* (e.g. to test warnings, or if you
have a collection of tests, only some of which error out), you can use the
`// compile-pass` header command to have the test runner instead
check that the test compiles successfully.

### Editing and updating the reference files

If you have changed the compiler's output intentionally, or you are
making a new test, you can pass `--bless` to the command you used to
run the tests. This will then copy over the files
from the build directory and use them as the new reference.

### Normalization

The normalization applied is aimed at eliminating output difference
between platforms, mainly about filenames:

- the test directory is replaced with `$DIR`
- all backslashes (`\`) are converted to forward slashes (`/`) (for Windows)
- all CR LF newlines are converted to LF

Sometimes these built-in normalizations are not enough. In such cases, you
may provide custom normalization rules using the header commands, e.g.

```
// normalize-stdout-test: "foo" -> "bar"
// normalize-stderr-32bit: "fn\(\) \(32 bits\)" -> "fn\(\) \($$PTR bits\)"
// normalize-stderr-64bit: "fn\(\) \(64 bits\)" -> "fn\(\) \($$PTR bits\)"
```

This tells the test, on 32-bit platforms, whenever the compiler writes
`fn() (32 bits)` to stderr, it should be normalized to read `fn() ($PTR bits)`
instead. Similar for 64-bit. The replacement is performed by regexes using
default regex flavor provided by `regex` crate.

The corresponding reference file will use the normalized output to test both
32-bit and 64-bit platforms:

```
...
   |
   = note: source type: fn() ($PTR bits)
   = note: target type: u16 (16 bits)
...
```

Please see `ui/transmute/main.rs` and `.stderr` for a concrete usage example.

Besides `normalize-stderr-32bit` and `-64bit`, one may use any target
information or stage supported by `ignore-X` here as well (e.g.
`normalize-stderr-windows` or simply `normalize-stderr-test` for unconditional
replacement).
