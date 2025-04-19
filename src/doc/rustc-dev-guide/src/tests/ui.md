# UI tests

<!-- toc -->

UI tests are a particular [test suite](compiletest.md#test-suites) of
compiletest.

## Introduction

The tests in [`tests/ui`] are a collection of general-purpose tests which
primarily focus on validating the console output of the compiler, but can be
used for many other purposes. For example, tests can also be configured to [run
the resulting program](#controlling-passfail-expectations) to verify its
behavior.

If you need to work with `#![no_std]` cross-compiling tests, consult the
[`minicore` test auxiliary](./minicore.md) chapter.

[`tests/ui`]: https://github.com/rust-lang/rust/blob/master/tests/ui

## General structure of a test

A test consists of a Rust source file located anywhere in the `tests/ui`
directory, but they should be placed in a suitable sub-directory. For example,
[`tests/ui/hello.rs`] is a basic hello-world test.

Compiletest will use `rustc` to compile the test, and compare the output against
the expected output which is stored in a `.stdout` or `.stderr` file located
next to the test. See [Output comparison](#output-comparison) for more.

Additionally, errors and warnings should be annotated with comments within the
source file. See [Error annotations](#error-annotations) for more.

Compiletest [directives](directives.md) in the form of special comments prefixed
with `//@` control how the test is compiled and what the expected behavior is.

Tests are expected to fail to compile, since most tests are testing compiler
errors. You can change that behavior with a directive, see [Controlling
pass/fail expectations](#controlling-passfail-expectations).

By default, a test is built as an executable binary. If you need a different
crate type, you can use the `#![crate_type]` attribute to set it as needed.

[`tests/ui/hello.rs`]: https://github.com/rust-lang/rust/blob/master/tests/ui/hello.rs

## Output comparison

UI tests store the expected output from the compiler in `.stderr` and `.stdout`
snapshots next to the test. You normally generate these files with the `--bless`
CLI option, and then inspect them manually to verify they contain what you
expect.

The output is normalized to ignore unwanted differences, see the
[Normalization](#normalization) section. If the file is missing, then
compiletest expects the corresponding output to be empty.

There can be multiple stdout/stderr files. The general form is:

```text
*test-name*`.`*revision*`.`*compare_mode*`.`*extension*
```

- *test-name* cannot contain dots. This is so that the general form of test
  output filenames have a predictable form we can pattern match on in order to
  track stray test output files.
- *revision* is the [revision](#cfg-revisions) name. This is not included when
  not using revisions.
- *compare_mode* is the [compare mode](#compare-modes). This will only be
  checked when the given compare mode is active. If the file does not exist,
  then compiletest will check for a file without the compare mode.
- *extension* is the kind of output being checked:
  - `stderr` — compiler stderr
  - `stdout` — compiler stdout
  - `run.stderr` — stderr when running the test
  - `run.stdout` — stdout when running the test
  - `64bit.stderr` — compiler stderr with `stderr-per-bitwidth` directive on a
    64-bit target
  - `32bit.stderr` — compiler stderr with `stderr-per-bitwidth` directive on a
    32-bit target

A simple example would be `foo.stderr` next to a `foo.rs` test.
A more complex example would be `foo.my-revision.polonius.stderr`.

There are several [directives](directives.md) which will change how compiletest
will check for output files:

- `stderr-per-bitwidth` — checks separate output files based on the target
  pointer width. Consider using the `normalize-stderr` directive instead (see
  [Normalization](#normalization)).
- `dont-check-compiler-stderr` — Ignores stderr from the compiler.
- `dont-check-compiler-stdout` — Ignores stdout from the compiler.

UI tests run with `-Zdeduplicate-diagnostics=no` flag which disables rustc's
built-in diagnostic deduplication mechanism. This means you may see some
duplicate messages in the output. This helps illuminate situations where
duplicate diagnostics are being generated.

### Normalization

The compiler output is normalized to eliminate output difference between
platforms, mainly about filenames.

Compiletest makes the following replacements on the compiler output:

- The directory where the test is defined is replaced with `$DIR`. Example:
  `/path/to/rust/tests/ui/error-codes`
- The directory to the standard library source is replaced with `$SRC_DIR`.
  Example: `/path/to/rust/library`
- Line and column numbers for paths in `$SRC_DIR` are replaced with `LL:COL`.
  This helps ensure that changes to the layout of the standard library do not
  cause widespread changes to the `.stderr` files. Example:
  `$SRC_DIR/alloc/src/sync.rs:53:46`
- The base directory where the test's output goes is replaced with
  `$TEST_BUILD_DIR`. This only comes up in a few rare circumstances. Example:
  `/path/to/rust/build/x86_64-unknown-linux-gnu/test/ui`
- Tabs are replaced with `\t`.
- Backslashes (`\`) are converted to forward slashes (`/`) within paths (using a
  heuristic). This helps normalize differences with Windows-style paths.
- CRLF newlines are converted to LF.
- Error line annotations like `//~ ERROR some message` are removed.
- Various v0 and legacy symbol hashes are replaced with placeholders like
  `[HASH]` or `<SYMBOL_HASH>`.

Additionally, the compiler is run with the `-Z ui-testing` flag which causes
the compiler itself to apply some changes to the diagnostic output to make it
more suitable for UI testing.

For example, it will anonymize line numbers in the output (line numbers
prefixing each source line are replaced with `LL`). In extremely rare
situations, this mode can be disabled with the directive `//@
compile-flags: -Z ui-testing=no`.

Note: The line and column numbers for `-->` lines pointing to the test are *not*
normalized, and left as-is. This ensures that the compiler continues to point to
the correct location, and keeps the stderr files readable. Ideally all
line/column information would be retained, but small changes to the source
causes large diffs, and more frequent merge conflicts and test errors.

Sometimes these built-in normalizations are not enough. In such cases, you may
provide custom normalization rules using `normalize-*` directives, e.g.

```rust,ignore
//@ normalize-stdout: "foo" -> "bar"
//@ normalize-stderr: "foo" -> "bar"
//@ normalize-stderr-32bit: "fn\(\) \(32 bits\)" -> "fn\(\) \($$PTR bits\)"
//@ normalize-stderr-64bit: "fn\(\) \(64 bits\)" -> "fn\(\) \($$PTR bits\)"
```

This tells the test, on 32-bit platforms, whenever the compiler writes `fn() (32
bits)` to stderr, it should be normalized to read `fn() ($PTR bits)` instead.
Similar for 64-bit. The replacement is performed by regexes using default regex
flavor provided by `regex` crate.

The corresponding reference file will use the normalized output to test both
32-bit and 64-bit platforms:

```text
...
   |
   = note: source type: fn() ($PTR bits)
   = note: target type: u16 (16 bits)
...
```

Please see [`ui/transmute/main.rs`][mrs] and [`main.stderr`] for a concrete
usage example.

[mrs]: https://github.com/rust-lang/rust/blob/master/tests/ui/transmute/main.rs
[`main.stderr`]: https://github.com/rust-lang/rust/blob/master/tests/ui/transmute/main.stderr

## Error annotations

Error annotations specify the errors that the compiler is expected to emit. They
are "attached" to the line in source where the error is located.

```rust,ignore
fn main() {
    boom  //~ ERROR cannot find value `boom` in this scope [E0425]
}
```

Although UI tests have a `.stderr` file which contains the entire compiler
output, UI tests require that errors are also annotated within the source. This
redundancy helps avoid mistakes since the `.stderr` files are usually
auto-generated. It also helps to directly see where the error spans are expected
to point to by looking at one file instead of having to compare the `.stderr`
file with the source. Finally, they ensure that no additional unexpected errors
are generated.

They have several forms, but generally are a comment with the diagnostic level
(such as `ERROR`) and a substring of the expected error output. You don't have
to write out the entire message, just make sure to include the important part of
the message to make it self-documenting.

The error annotation needs to match with the line of the diagnostic. There are
several ways to match the message with the line (see the examples below):

* `~`: Associates the error level and message with the *current* line
* `~^`: Associates the error level and message with the *previous* error
  annotation line. Each caret (`^`) that you add adds a line to this, so `~^^^`
  is three lines above the error annotation line.
* `~|`: Associates the error level and message with the *same* line as the
  *previous comment*. This is more convenient than using multiple carets when
  there are multiple messages associated with the same line.
* `~v`: Associates the error level and message with the *next* error
  annotation line. Each symbol (`v`) that you add adds a line to this, so `~vvv`
  is three lines below the error annotation line.
* `~?`: Used to match error levels and messages with errors not having line
  information. These can be placed on any line in the test file, but are
  conventionally placed at the end.

Example:

```rust,ignore
let _ = same_line; //~ ERROR undeclared variable
fn meow(_: [u8]) {}
//~^ ERROR unsized
//~| ERROR anonymous parameters
```

The space character between `//~` (or other variants) and the subsequent text is
negligible (i.e. there is no semantic difference between `//~ ERROR` and
`//~ERROR` although the former is more common in the codebase).

### Error annotation examples

Here are examples of error annotations on different lines of UI test source.

#### Positioned on error line

Use the `//~ ERROR` idiom:

```rust,ignore
fn main() {
    let x = (1, 2, 3);
    match x {
        (_a, _x @ ..) => {} //~ ERROR `_x @` is not allowed in a tuple
        _ => {}
    }
}
```

#### Positioned below error line

Use the `//~^` idiom with number of carets in the string to indicate the number
of lines above. In the example below, the error line is four lines above the
error annotation line so four carets are included in the annotation.

```rust,ignore
fn main() {
    let x = (1, 2, 3);
    match x {
        (_a, _x @ ..) => {}  // <- the error is on this line
        _ => {}
    }
}
//~^^^^ ERROR `_x @` is not allowed in a tuple
```

#### Use same error line as defined on error annotation line above

Use the `//~|` idiom to define the same error line as the error annotation
line above:

```rust,ignore
struct Binder(i32, i32, i32);

fn main() {
    let x = Binder(1, 2, 3);
    match x {
        Binder(_a, _x @ ..) => {}  // <- the error is on this line
        _ => {}
    }
}
//~^^^^ ERROR `_x @` is not allowed in a tuple struct
//~| ERROR this pattern has 1 field, but the corresponding tuple struct has 3 fields [E0023]
```

#### Positioned above error line

Use the `//~v` idiom with number of v's in the string to indicate the number
of lines below. This is typically used in lexer or parser tests matching on errors like unclosed
delimiter or unclosed literal happening at the end of file.

```rust,ignore
// ignore-tidy-trailing-newlines
//~v ERROR this file contains an unclosed delimiter
fn main((ؼ
```

#### Error without line information

Use `//~?` to match an error without line information.
`//~?` is precise and will not match errors if their line information is available.
It should be preferred to using `error-pattern`, which is imprecise and non-exhaustive.

```rust,ignore
//@ compile-flags: --print yyyy

//~? ERROR unknown print request: `yyyy`
```

### `error-pattern`

The `error-pattern` [directive](directives.md) can be used for runtime messages, which don't
have a specific span, or in exceptional cases, for compile time messages.

Let's think about this test:

```rust,ignore
fn main() {
    let a: *const [_] = &[1, 2, 3];
    unsafe {
        let _b = (*a)[3];
    }
}
```

We want to ensure this shows "index out of bounds", but we cannot use the `ERROR`
annotation since the runtime error doesn't have any span. Then it's time to use the
`error-pattern` directive:

```rust,ignore
//@ error-pattern: index out of bounds
fn main() {
    let a: *const [_] = &[1, 2, 3];
    unsafe {
        let _b = (*a)[3];
    }
}
```

Use of `error-pattern` is not recommended in general.

For strict testing of compile time output, try to use the line annotations `//~` as much as
possible, including `//~?` annotations for diagnostics without spans.

If the compile time output is target dependent or too verbose, use directive
`//@ dont-require-annotations: <diagnostic-kind>` to make the line annotation checking
non-exhaustive.
Some of the compiler messages can stay uncovered by annotations in this mode.

For checking runtime output, `//@ check-run-results` may be preferable.

Only use `error-pattern` if none of the above works.

Line annotations `//~` are still checked in tests using `error-pattern`.
In exceptional cases, use `//@ compile-flags: --error-format=human` to opt out of these checks.

### Diagnostic kinds (error levels)

The diagnostic kinds that you can have are:

- `ERROR`
- `WARN` (or `WARNING`)
- `NOTE`
- `HELP`
- `SUGGESTION`

The `SUGGESTION` kind is used for specifying what the expected replacement text
should be for a diagnostic suggestion.

`ERROR` and `WARN` kinds are required to be exhaustively covered by line annotations
`//~` by default.

Other kinds only need to be line-annotated if at least one annotation of that kind appears
in the test file. For example, one `//~ NOTE` will also require all other `//~ NOTE`s in the file
to be written out explicitly.

Use directive `//@ dont-require-annotations` to opt out of exhaustive annotations.
E.g. use `//@ dont-require-annotations: NOTE` to annotate notes selectively.
Avoid using this directive for `ERROR`s and `WARN`ings, unless there's a serious reason, like
target-dependent compiler output.

Missing diagnostic kinds (`//~ message`) are currently accepted, but are being phased away.
They will match any compiler output kind, but will not force exhaustive annotations for that kind.
Prefer explicit kind and `//@ dont-require-annotations` to achieve the same effect.

UI tests use the `-A unused` flag by default to ignore all unused warnings, as
unused warnings are usually not the focus of a test. However, simple code
samples often have unused warnings. If the test is specifically testing an
unused warning, just add the appropriate `#![warn(unused)]` attribute as needed.

### `cfg` revisions

When using [revisions](compiletest.md#revisions), different messages can be
conditionally checked based on the current revision. This is done by placing the
revision cfg name in brackets like this:

```rust,ignore
//@ edition:2018
//@ revisions: mir thir
//@[thir] compile-flags: -Z thir-unsafeck

async unsafe fn f() {}

async fn g() {
    f(); //~ ERROR call to unsafe function is unsafe
}

fn main() {
    f(); //[mir]~ ERROR call to unsafe function is unsafe
}
```

In this example, the second error message is only emitted in the `mir` revision.
The `thir` revision only emits the first error.

If the `cfg` causes the compiler to emit different output, then a test can have
multiple `.stderr` files for the different outputs. In the example above, there
would be a `.mir.stderr` and `.thir.stderr` file with the different outputs of
the different revisions.

> Note: cfg revisions also work inside the source code with `#[cfg]` attributes.
>
> By convention, the `FALSE` cfg is used to have an always-false config.

## Controlling pass/fail expectations

By default, a UI test is expected to **generate a compile error** because most
of the tests are checking for invalid input and error diagnostics. However, you
can also make UI tests where compilation is expected to succeed, and you can
even run the resulting program. Just add one of the following
[directives](directives.md):

- Pass directives:
  - `//@ check-pass` — compilation should succeed but skip codegen
    (which is expensive and isn't supposed to fail in most cases).
  - `//@ build-pass` — compilation and linking should succeed but do
    not run the resulting binary.
  - `//@ run-pass` — compilation should succeed and running the resulting
    binary should also succeed.
- Fail directives:
  - `//@ check-fail` — compilation should fail (the codegen phase is skipped).
    This is the default for UI tests.
  - `//@ build-fail` — compilation should fail during the codegen phase.
    This will run `rustc` twice, once to verify that it compiles successfully
    without the codegen phase, then a second time the full compile should
    fail.
  - `//@ run-fail` — compilation should succeed, but running the resulting
    binary should fail.

For `run-pass` and `run-fail` tests, by default the output of the program itself
is not checked.

If you want to check the output of running the program, include the
`check-run-results` directive. This will check for a `.run.stderr` and
`.run.stdout` files to compare against the actual output of the program.

Tests with the `*-pass` directives can be overridden with the `--pass`
command-line option:

```sh
./x test tests/ui --pass check
```

The `--pass` option only affects UI tests. Using `--pass check` can run the UI
test suite much faster (roughly twice as fast on my system), though obviously
not exercising as much.

The `ignore-pass` directive can be used to ignore the `--pass` CLI flag if the
test won't work properly with that override.


## Known bugs

The `known-bug` directive may be used for tests that demonstrate a known bug
that has not yet been fixed. Adding tests for known bugs is helpful for several
reasons, including:

1. Maintaining a functional test that can be conveniently reused when the bug is
   fixed.
2. Providing a sentinel that will fail if the bug is incidentally fixed. This
   can alert the developer so they know that the associated issue has been fixed
   and can possibly be closed.

This directive takes comma-separated issue numbers as arguments, or `"unknown"`:

- `//@ known-bug: #123, #456` (when the issues are on rust-lang/rust)
- `//@ known-bug: rust-lang/chalk#123456`
  (allows arbitrary text before the `#`, which is useful when the issue is on another repo)
- `//@ known-bug: unknown`
  (when there is no known issue yet; preferrably open one if it does not already exist)

Do not include [error annotations](#error-annotations) in a test with
`known-bug`. The test should still include other normal directives and
stdout/stderr files.


## Test organization

When deciding where to place a test file, please try to find a subdirectory that
best matches what you are trying to exercise. Do your best to keep things
organized. Admittedly it can be difficult as some tests can overlap different
categories, and the existing layout may not fit well.

Name the test by a concise description of what the test is checking. Avoid
including the issue number in the test name. See [best
practices](best-practices.md) for a more in-depth discussion of this.

Ideally, the test should be added to a directory that helps identify what piece
of code is being tested here (e.g.,
`tests/ui/borrowck/reject-move-out-of-borrow-via-pat.rs`)

When writing a new feature, you may want to **create a subdirectory to store
your tests**. For example, if you are implementing RFC 1234 ("Widgets"), then it
might make sense to put the tests in a directory like
`tests/ui/rfc1234-widgets/`.

In other cases, there may already be a suitable directory.

Over time, the [`tests/ui`] directory has grown very fast. There is a check in
[tidy](intro.md#tidy) that will ensure none of the subdirectories has more than
1000 entries. Having too many files causes problems because it isn't editor/IDE
friendly and the GitHub UI won't show more than 1000 entries. However, since
`tests/ui` (UI test root directory) and `tests/ui/issues` directories have more
than 1000 entries, we set a different limit for those directories. So, please
avoid putting a new test there and try to find a more relevant place.

For example, if your test is related to closures, you should put it in
`tests/ui/closures`. When you reach the limit, you could increase it by tweaking
[here][ui test tidy].

[ui test tidy]: https://github.com/rust-lang/rust/blob/master/src/tools/tidy/src/ui_tests.rs

## Rustfix tests

UI tests can validate that diagnostic suggestions apply correctly and that the
resulting changes compile correctly. This can be done with the `run-rustfix`
directive:

```rust,ignore
//@ run-rustfix
//@ check-pass
#![crate_type = "lib"]

pub struct not_camel_case {}
//~^ WARN `not_camel_case` should have an upper camel case name
//~| HELP convert the identifier to upper camel case
//~| SUGGESTION NotCamelCase
```

Rustfix tests should have a file with the `.fixed` extension which contains the
source file after the suggestion has been applied.

- When the test is run, compiletest first checks that the correct lint/warning
  is generated.
- Then, it applies the suggestion and compares against `.fixed` (they must
  match).
- Finally, the fixed source is compiled, and this compilation is required to
  succeed.

Usually when creating a rustfix test you will generate the `.fixed` file
automatically with the `x test --bless` option.

The `run-rustfix` directive will cause *all* suggestions to be applied, even if
they are not [`MachineApplicable`](../diagnostics.md#suggestions). If this is a
problem, then you can add the `rustfix-only-machine-applicable` directive in
addition to `run-rustfix`. This should be used if there is a mixture of
different suggestion levels, and some of the non-machine-applicable ones do not
apply cleanly.


## Compare modes

[Compare modes](compiletest.md#compare-modes) can be used to run all tests with
different flags from what they are normally compiled with. In some cases, this
might result in different output from the compiler. To support this, different
output files can be saved which contain the output based on the compare mode.

For example, when using the Polonius mode, a test `foo.rs` will first look for
expected output in `foo.polonius.stderr`, falling back to the usual `foo.stderr`
if not found. This is useful as different modes can sometimes result in
different diagnostics and behavior. This can help track which tests have
differences between the modes, and to visually inspect those diagnostic
differences.

If in the rare case you encounter a test that has different behavior, you can
run something like the following to generate the alternate stderr file:

```sh
./x test tests/ui --compare-mode=polonius --bless
```

Currently none of the compare modes are checked in CI for UI tests.

## `rustc_*` TEST attributes

The compiler defines several perma-unstable `#[rustc_*]` attributes gated behind
the internal feature `rustc_attrs` that dump extra compiler-internal
information. See the corresponding subsection in [compiler debugging] for more
details.

They can be used in tests to more precisely, legibly and easily test internal
compiler state in cases where it would otherwise be very hard to do the same
with "user-facing" Rust alone. Indeed, one could say that this slightly abuses
the term "UI" (*user* interface) and turns such UI tests from black-box tests
into white-box ones. Use them carefully and sparingly.

[compiler debugging]: ../compiler-debugging.md#rustc_-test-attributes

## UI test mode preset lint levels

By default, test suites under UI test mode (`tests/ui`, `tests/ui-fulldeps`,
but not `tests/rustdoc-ui`) will specify

- `-A unused`
- `-A internal_features`

If:

- The ui test's pass mode is below `run` (i.e. check or build).
- No compare modes are specified.

Since they can be very noisy in ui tests.

You can override them with `compile-flags` lint level flags or
in-source lint level attributes as required.

Note that the `rustfix` version will *not* have `-A unused` passed,
meaning that you may have to `#[allow(unused)]` to suppress `unused`
lints on the rustfix'd file (because we might be testing rustfix
on `unused` lints themselves).
