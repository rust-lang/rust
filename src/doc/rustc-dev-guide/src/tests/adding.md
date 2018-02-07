# Adding new tests

**In general, we expect every PR that fixes a bug in rustc to come
accompanied with a regression test of some kind.** This test should
fail in master but pass after the PR. These tests are really useful
for preventing us from repeating the mistakes of the past.

To add a new test, the first thing you generally do is to create a
file, typically a Rust source file. Test files have a particular
structure:

- They always begin with the copyright notice;
- then they should have some kind of
  [comment explaining what the test is about](#explanatory_comment);
- next, they can have one or more [header commands](#header_commands), which are special
  comments that the test interpreter knows how to interpret.
- finally, they have the Rust source. This may have various [error
  annotations](#error_annotations) which indicate expected compilation errors or
  warnings.

Depending on the test suite, there may be some other details to be aware of:
  - For [the `ui` test suite](#ui), you need to generate reference output files.
  
## Naming your test

We have not traditionally had a lot of structure in the names of
tests.  Moreover, for a long time, the rustc test runner did not
support subdirectories (it now does), so test suites like
`src/test/run-pass` have a huge mess of files in them. This is not
considered an ideal setup.

For regression tests -- basically, some random snippet of code that
came in from the internet -- we often just name the test after the
issue. For example, `src/test/run-pass/issue-1234.rs`. If possible,
though, it is better if you can put the test into a directory that
helps identify what piece of code is being tested here (e.g.,
`borrowck/issue-12345.rs` is much better), or perhaps give it a more
meaningful name. Still, **do include the issue number somewhere**.

When writing a new feature, **create a subdirectory to store your
tests**. For example, if you are implementing RFC 1234 ("Widgets"),
then it might make sense to put the tests in directories like:

- `src/test/ui/rfc1234-widgets/` 
- `src/test/run-pass/rfc1234-widgets/` 
- etc

In other cases, there may already be a suitable directory. (The proper
directory structure to use is actually an area of active debate.)

<a name=explanatory_comment>

## Comment explaining what the test is about

When you create a test file, **include a comment summarizing the point
of the test immediately after the copyright notice**. This should
highlight which parts of the test are more important, and what the bug
was that the test is fixing.  Citing an issue number is often very
helpful.

This comment doesn't have to be super extensive. Just something like
"Regression test for #18060: match arms were matching in the wrong
order."  might already be enough.

These comments are very useful to others later on when your test
breaks, since they often can highlight what the problem is. They are
also useful if for some reason the tests need to be refactored, since
they let others know which parts of the test were important (often a
test must be rewritten because it no longer tests what is was meant to
test, and then it's useful to know what it *was* meant to test
exactly).

<a name=header_commands>

## Header commands: configuring rustc

Header commands are special comments that the test runner knows how to
interpret.  They must appear before the Rust source in the test. They
are normally put after the short comment that explains the point of
this test. For example, this test uses the `// compile-flags` command
to specify a custom flag to give to rustc when the test is compiled:

```rust
// Copyright 2017 The Rust Project Developers. blah blah blah.
// ...
// except according to those terms.

// Test the behavior of `0 - 1` when overflow checks are disabled.

// compile-flags: -Coverflow-checks=off

fn main() {
    let x = 0 - 1;
    ...
}
```

### Ignoring tests

These are used to ignore the test in some situations, which means the test won't
be compiled or run.

* `ignore-X` where `X` is a target detail or stage will ignore the test accordingly (see below)
* `ignore-pretty` will not compile the pretty-printed test (this is done to test the pretty-printer, but might not always work)
* `ignore-test` always ignores the test
* `ignore-lldb` and `ignore-gdb` will skip a debuginfo test on that debugger.

Some examples of `X` in `ignore-X`:

* Architecture: `aarch64`, `arm`, `asmjs`, `mips`, `wasm32`, `x86_64`, `x86`, ...
* OS: `android`, `emscripten`, `freebsd`, `ios`, `linux`, `macos`, `windows`, ...
* Environment (fourth word of the target triple): `gnu`, `msvc`, `musl`.
* Pointer width: `32bit`, `64bit`.
* Stage: `stage0`, `stage1`, `stage2`.

### Other Header Commands

Here is a list of other header commands. This list is not
exhaustive. Header commands can generally be found by browsing the
`TestProps` structure found in [`header.rs`] from the compiletest
source.

* `min-{gdb,lldb}-version`
* `min-llvm-version`
* `must-compile-successfully` for UI tests, indicates that the test is supposed
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

[`header.rs`]: https://github.com/rust-lang/rust/tree/master/src/tools/compiletest/src/header.rs

<a name="error_annotations">

## Error annotations

Error annotations specify the errors that the compiler is expected to
emit. They are "attached" to the line in source where the error is
located.

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

<a name="ui">

## Guide to the UI tests

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
`// must-compile-successfully` header command to have the test runner instead
check that the test compiles successfully.

### Editing and updating the reference files

If you have changed the compiler's output intentionally, or you are
making a new test, you can use the script `ui/update-references.sh` to
update the references. When you run the test framework, it will report
various errors: in those errors is a command you can use to run the
`ui/update-references.sh` script, which will then copy over the files
from the build directory and use them as the new reference. You can
also just run `ui/update-all-references.sh`. In both cases, you can run
the script with `--help` to get a help message.

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
