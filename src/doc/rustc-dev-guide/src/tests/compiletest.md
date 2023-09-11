# Compiletest

<!-- toc -->

## Introduction

`compiletest` is the main test harness of the Rust test suite.
It allows test authors to organize large numbers of tests
(the Rust compiler has many thousands),
efficient test execution (parallel execution is supported),
and allows the test author to configure behavior and expected results of both
individual and groups of tests.

> NOTE:
> For macOS users, `SIP` (System Integrity Protection) [may consistently
> check the compiled binary by sending network requests to Apple][zulip],
> so you may get a huge performance degradation when running tests.
>
> You can resolve it by tweaking the following settings:
> `Privacy & Security -> Developer Tools -> Add Terminal (Or VsCode, etc.)`.

[zulip]: https://rust-lang.zulipchat.com/#narrow/stream/182449-t-compiler.2Fhelp/topic/.E2.9C.94.20Is.20there.20any.20performance.20issue.20for.20MacOS.3F

`compiletest` may check test code for success, for runtime failure,
or for compile-time failure.
Tests are typically organized as a Rust source file with annotations in
comments before and/or within the test code.
These comments serve to direct `compiletest` on if or how to run the test,
what behavior to expect, and more.
See [header commands](headers.md) and the test suite documentation below
for more details on these annotations.

See the [Adding new tests](adding.md) chapter for a tutorial on creating a new
test, and the [Running tests](running.md) chapter on how to run the test
suite.

Compiletest itself tries to avoid running tests when the artifacts
that are involved (mainly the compiler) haven't changed. You can use
`x test --test-args --force-rerun` to rerun a test even when none of the
inputs have changed.

## Test suites

All of the tests are in the [`tests`] directory.
The tests are organized into "suites", with each suite in a separate subdirectory.
Each test suite behaves a little differently, with different compiler behavior
and different checks for correctness.
For example, the [`tests/incremental`] directory contains tests for
incremental compilation.
The various suites are defined in [`src/tools/compiletest/src/common.rs`] in
the `pub enum Mode` declaration.

The following test suites are available, with links for more information:

- [`ui`](ui.md) — tests that check the stdout/stderr from the compilation
  and/or running the resulting executable
- `ui-fulldeps` — `ui` tests which require a linkable build of `rustc` (such
  as using `extern crate rustc_span;` or used as a plugin)
- [`pretty`](#pretty-printer-tests) — tests for pretty printing
- [`incremental`](#incremental-tests) — tests incremental compilation behavior
- [`debuginfo`](#debuginfo-tests) — tests for debuginfo generation running debuggers
- [`codegen`](#codegen-tests) — tests for code generation
- [`codegen-units`](#codegen-units-tests) — tests for codegen unit partitioning
- [`assembly`](#assembly-tests) — verifies assembly output
- [`mir-opt`](#mir-opt-tests) — tests for MIR generation
- [`run-make`](#run-make-tests) — general purpose tests using a Makefile
- `run-make-fulldeps` — `run-make` tests which require a linkable build of `rustc`,
  or the rust demangler
- [`run-pass-valgrind`](#valgrind-tests) — tests run with Valgrind
- [`coverage-map`](#coverage-tests) - tests for coverage maps produced by
  coverage instrumentation
- [`run-coverage`](#coverage-tests) - tests that run an instrumented program
  and check its coverage report
- [`run-coverage-rustdoc`](#coverage-tests) - coverage tests that also run
  instrumented doctests
- [Rustdoc tests](../rustdoc.md#tests):
    - `rustdoc` — tests for rustdoc, making sure that the generated files
      contain the expected documentation.
    - `rustdoc-gui` — tests for rustdoc's GUI using a web browser.
    - `rustdoc-js` — tests to ensure the rustdoc search is working as expected.
    - `rustdoc-js-std` — tests to ensure the rustdoc search is working as expected
      (run specifically on the std docs).
    - `rustdoc-json` — tests on the JSON output of rustdoc.
    - `rustdoc-ui` — tests on the terminal output of rustdoc.

[`tests`]: https://github.com/rust-lang/rust/blob/master/tests
[`src/tools/compiletest/src/common.rs`]: https://github.com/rust-lang/rust/tree/master/src/tools/compiletest/src/common.rs

### Pretty-printer tests

The tests in [`tests/pretty`] exercise the "pretty-printing" functionality of `rustc`.
The `-Z unpretty` CLI option for `rustc` causes it to translate the input source
into various different formats, such as the Rust source after macro expansion.

The pretty-printer tests have several [header commands](headers.md) described below.
These commands can significantly change the behavior of the test, but the
default behavior without any commands is to:

1. Run `rustc -Zunpretty=normal` on the source file
2. Run `rustc -Zunpretty=normal` on the output of the previous step
3. The output of the previous two steps should be the same.
4. Run `rustc -Zno-codegen` on the output to make sure that it can type check
   (this is similar to running `cargo check`)

If any of the commands above fail, then the test fails.

The header commands for pretty-printing tests are:

* `pretty-mode` specifies the mode pretty-print tests should run in
  (that is, the argument to `-Zunpretty`).
  The default is `normal` if not specified.
* `pretty-compare-only` causes a pretty test to only compare the pretty-printed output
  (stopping after step 3 from above).
  It will not try to compile the expanded output to type check it.
  This is needed for a pretty-mode that does not expand to valid
  Rust, or for other situations where the expanded output cannot be compiled.
* `pretty-expanded` allows a pretty test to also check that the expanded
  output can be type checked.
  That is, after the steps above, it does two more steps:

  > 5. Run `rustc -Zunpretty=expanded` on the original source
  > 6. Run `rustc -Zno-codegen` on the expanded output to make sure that it can type check

  This is needed because not all code can be compiled after being expanded.
  Pretty tests should specify this if they can.
  An example where this cannot be used is if the test includes `println!`.
  That macro expands to reference private internal functions of the standard
  library that cannot be called directly without the `fmt_internals` feature
  gate.

  More history about this may be found in
  [#23616](https://github.com/rust-lang/rust/issues/23616#issuecomment-484999901).
* `pp-exact` is used to ensure a pretty-print test results in specific output.
  If specified without a value, then it means the pretty-print output should
  match the original source.
  If specified with a value, as in `// pp-exact:foo.pp`,
  it will ensure that the pretty-printed output matches the contents of the given file.
  Otherwise, if `pp-exact` is not specified, then the pretty-printed output
  will be pretty-printed one more time, and the output of the two
  pretty-printing rounds will be compared to ensure that the pretty-printed
  output converges to a steady state.

[`tests/pretty`]: https://github.com/rust-lang/rust/tree/master/tests/pretty

### Incremental tests

The tests in [`tests/incremental`] exercise incremental compilation.
They use [revision headers](#revisions) to tell compiletest to run the
compiler in a series of steps.
Compiletest starts with an empty directory with the `-C incremental` flag, and
then runs the compiler for each revision, reusing the incremental results from
previous steps.
The revisions should start with:

* `rpass` — the test should compile and run successfully
* `rfail` — the test should compile successfully, but the executable should fail to run
* `cfail` — the test should fail to compile

To make the revisions unique, you should add a suffix like `rpass1` and `rpass2`.

To simulate changing the source, compiletest also passes a `--cfg` flag with
the current revision name.
For example, this will run twice, simulating changing a function:

```rust,ignore
// revisions: rpass1 rpass2

#[cfg(rpass1)]
fn foo() {
    println!("one");
}

#[cfg(rpass2)]
fn foo() {
    println!("two");
}

fn main() { foo(); }
```

`cfail` tests support the `forbid-output` header to specify that a certain
substring must not appear anywhere in the compiler output.
This can be useful to ensure certain errors do not appear, but this can be
fragile as error messages change over time, and a test may no longer be
checking the right thing but will still pass.

`cfail` tests support the `should-ice` header to specify that a test should
cause an Internal Compiler Error (ICE).
This is a highly specialized header to check that the incremental cache
continues to work after an ICE.

[`tests/incremental`]: https://github.com/rust-lang/rust/tree/master/tests/incremental


### Debuginfo tests

The tests in [`tests/debuginfo`] test debuginfo generation.
They build a program, launch a debugger, and issue commands to the debugger.
A single test can work with cdb, gdb, and lldb.

Most tests should have the `// compile-flags: -g` header or something similar
to generate the appropriate debuginfo.

To set a breakpoint on a line, add a `// #break` comment on the line.

The debuginfo tests consist of a series of debugger commands along with
"check" lines which specify output that is expected from the debugger.

The commands are comments of the form `// $DEBUGGER-command:$COMMAND` where
`$DEBUGGER` is the debugger being used and `$COMMAND` is the debugger command
to execute.
The debugger values can be:

* `cdb`
* `gdb`
* `gdbg` — GDB without Rust support (versions older than 7.11)
* `gdbr` — GDB with Rust support
* `lldb`
* `lldbg` — LLDB without Rust support
* `lldbr` — LLDB with Rust support (this no longer exists)

The command to check the output are of the form `// $DEBUGGER-check:$OUTPUT`
where `$OUTPUT` is the output to expect.

For example, the following will build the test, start the debugger, set a
breakpoint, launch the program, inspect a value, and check what the debugger
prints:

```rust,ignore
// compile-flags: -g

// lldb-command: run
// lldb-command: print foo
// lldb-check: $0 = 123

fn main() {
    let foo = 123;
    b(); // #break
}

fn b() {}
```

The following [header commands](headers.md) are available to disable a
test based on the debugger currently being used:

* `min-cdb-version: 10.0.18317.1001` — ignores the test if the version of cdb
  is below the given version
* `min-gdb-version: 8.2` — ignores the test if the version of gdb is below the
  given version
* `ignore-gdb-version: 9.2` — ignores the test if the version of gdb is equal
  to the given version
* `ignore-gdb-version: 7.11.90 - 8.0.9` — ignores the test if the version of
  gdb is in a range (inclusive)
* `min-lldb-version: 310` — ignores the test if the version of lldb is below
  the given version
* `rust-lldb` — ignores the test if lldb is not contain the Rust plugin.
  NOTE: The "Rust" version of LLDB doesn't exist anymore, so this will always be ignored.
  This should probably be removed.

[`tests/debuginfo`]: https://github.com/rust-lang/rust/tree/master/tests/debuginfo


### Codegen tests

The tests in [`tests/codegen`] test LLVM code generation.
They compile the test with the `--emit=llvm-ir` flag to emit LLVM IR.
They then run the LLVM [FileCheck] tool.
The test is annotated with various `// CHECK` comments to check the generated code.
See the FileCheck documentation for a tutorial and more information.

See also the [assembly tests](#assembly-tests) for a similar set of tests.

[`tests/codegen`]: https://github.com/rust-lang/rust/tree/master/tests/codegen
[FileCheck]: https://llvm.org/docs/CommandGuide/FileCheck.html


### Assembly tests

The tests in [`tests/assembly`] test LLVM assembly output.
They compile the test with the `--emit=asm` flag to emit a `.s` file with the
assembly output.
They then run the LLVM [FileCheck] tool.

Each test should be annotated with the `// assembly-output:` header
with a value of either `emit-asm` or `ptx-linker` to indicate
the type of assembly output.

Then, they should be annotated with various `// CHECK` comments to check the
assembly output.
See the FileCheck documentation for a tutorial and more information.

See also the [codegen tests](#codegen-tests) for a similar set of tests.

[`tests/assembly`]: https://github.com/rust-lang/rust/tree/master/tests/assembly


### Codegen-units tests

The tests in [`tests/codegen-units`] test the
[monomorphization](../backend/monomorph.md) collector and CGU partitioning.

These tests work by running `rustc` with a flag to print the result of the
monomorphization collection pass, and then special annotations in the file are
used to compare against that.

Each test should be annotated with the `// compile-flags:-Zprint-mono-items=VAL`
header with the appropriate VAL to instruct `rustc` to print the
monomorphization information.

Then, the test should be annotated with comments of the form `//~ MONO_ITEM name`
where `name` is the monomorphized string printed by rustc like `fn <u32 as Trait>::foo`.

To check for CGU partitioning, a comment of the form `//~ MONO_ITEM name @@ cgu`
where `cgu` is a space separated list of the CGU names and the linkage
information in brackets.
For example: `//~ MONO_ITEM static function::FOO @@ statics[Internal]`

[`tests/codegen-units`]: https://github.com/rust-lang/rust/tree/master/tests/codegen-units


### Mir-opt tests

The tests in [`tests/mir-opt`] check parts of the generated MIR to make
sure it is generated correctly and is doing the expected optimizations.
Check out the [MIR Optimizations](../mir/optimizations.md) chapter for more.

Compiletest will build the test with several flags to dump the MIR output and
set a baseline for optimizations:

* `-Copt-level=1`
* `-Zdump-mir=all`
* `-Zmir-opt-level=4`
* `-Zvalidate-mir`
* `-Zdump-mir-exclude-pass-number`

The test should be annotated with `// EMIT_MIR` comments that specify files that
will contain the expected MIR output.
You can use `x test --bless` to create the initial expected files.

There are several forms the `EMIT_MIR` comment can take:

* `// EMIT_MIR $MIR_PATH.mir` — This will check that the given filename
  matches the exact output from the MIR dump.
  For example, `my_test.main.SimplifyCfg-elaborate-drops.after.mir` will load
  that file from the test directory, and compare it against the dump from
  rustc.

  Checking the "after" file (which is after optimization) is useful if you are
  interested in the final state after an optimization.
  Some rare cases may want to use the "before" file for completeness.

* `// EMIT_MIR $MIR_PATH.diff` — where `$MIR_PATH` is the filename of the MIR
  dump, such as `my_test_name.my_function.EarlyOtherwiseBranch`.
  Compiletest will diff the `.before.mir` and `.after.mir` files, and compare
  the diff output to the expected `.diff` file from the `EMIT_MIR` comment.

  This is useful if you want to see how an optimization changes the MIR.

* `// EMIT_MIR $MIR_PATH.dot` or `$MIR_PATH.html` — These are special cases
  for other MIR outputs (via `-Z dump-mir-graphviz` and `-Z dump-mir-spanview`)
  that will check that the output matches the given file.

By default 32 bit and 64 bit targets use the same dump files, which can be
problematic in the presence of pointers in constants or other bit width
dependent things. In that case you can add `// EMIT_MIR_FOR_EACH_BIT_WIDTH` to
your test, causing separate files to be generated for 32bit and 64bit systems.

[`tests/mir-opt`]: https://github.com/rust-lang/rust/tree/master/tests/mir-opt


### `run-make` tests

The tests in [`tests/run-make`] are general-purpose tests using Makefiles
which provide the ultimate in flexibility.
These should be used as a last resort.
If possible, you should use one of the other test suites.
If there is some minor feature missing which you need for your test,
consider extending compiletest to add a header command for what you need.
However, if running a bunch of commands is really what you need,
`run-make` is here to the rescue!

Each test should be in a separate directory with a `Makefile` indicating the
commands to run.
There is a [`tools.mk`] Makefile which you can include which provides a bunch of
utilities to make it easier to run commands and compare outputs.
Take a look at some of the other tests for some examples on how to get started.

[`tools.mk`]: https://github.com/rust-lang/rust/blob/master/tests/run-make/tools.mk
[`tests/run-make`]: https://github.com/rust-lang/rust/tree/master/tests/run-make


### Valgrind tests

The tests in [`tests/run-pass-valgrind`] are for use with [Valgrind].
These are currently vestigial, as Valgrind is no longer used in CI.
These may be removed in the future.

[Valgrind]: https://valgrind.org/
[`tests/run-pass-valgrind`]: https://github.com/rust-lang/rust/tree/master/tests/run-pass-valgrind


### Coverage tests

The tests in [`tests/coverage-map`] test the mappings between source code
regions and coverage counters that are emitted by LLVM.
They compile the test with `--emit=llvm-ir`,
then use a custom tool ([`src/tools/coverage-dump`])
to extract and pretty-print the coverage mappings embedded in the IR.
These tests don't require the profiler runtime, so they run in PR CI jobs and
are easy to run/bless locally.

These coverage map tests can be sensitive to changes in MIR lowering or MIR
optimizations, producing mappings that are different but produce identical
coverage reports.

As a rule of thumb, any PR that doesn't change coverage-specific
code should **feel free to re-bless** the `coverage-map` tests as necessary,
without worrying about the actual changes, as long as the `run-coverage` tests
still pass.

---

The tests in [`tests/run-coverage`] perform an end-to-end test of coverage reporting.
They compile a test program with coverage instrumentation, run that program to
produce raw coverage data, and then use LLVM tools to process that data into a
human-readable code coverage report.

Instrumented binaries need to be linked against the LLVM profiler runtime,
so `run-coverage` tests are **automatically skipped**
unless the profiler runtime is enabled in `config.toml`:

```toml
# config.toml
[build]
profiler = true
```

This also means that they typically don't run in PR CI jobs,
though they do run in the full set of CI jobs used for merging.

The tests in [`tests/run-coverage-rustdoc`] also run instrumented doctests and
include them in the coverage report. This avoids having to build rustdoc when
only running the main `run-coverage` suite.

[`tests/coverage-map`]: https://github.com/rust-lang/rust/tree/master/tests/coverage-map
[`src/tools/coverage-dump`]: https://github.com/rust-lang/rust/tree/master/src/tools/coverage-dump
[`tests/run-coverage`]: https://github.com/rust-lang/rust/tree/master/tests/run-coverage
[`tests/run-coverage-rustdoc`]: https://github.com/rust-lang/rust/tree/master/tests/run-coverage-rustdoc


## Building auxiliary crates

It is common that some tests require additional auxiliary crates to be compiled.
There are two [headers](headers.md) to assist with that:

* `aux-build`
* `aux-crate`

`aux-build` will build a separate crate from the named source file.
The source file should be in a directory called `auxiliary` beside the test file.

```rust,ignore
// aux-build: my-helper.rs

extern crate my_helper;
// ... You can use my_helper.
```

The aux crate will be built as a dylib if possible (unless on a platform that
does not support them, or the `no-prefer-dynamic` header is specified in the
aux file).
The `-L` flag is used to find the extern crates.

`aux-crate` is very similar to `aux-build`; however, it uses the `--extern`
flag to link to the extern crate.
That allows you to specify the additional syntax of the `--extern` flag, such
as renaming a dependency.
For example, `// aux-crate:foo=bar.rs` will compile `auxiliary/bar.rs` and
make it available under then name `foo` within the test.
This is similar to how Cargo does dependency renaming.

### Auxiliary proc-macro

If you want a proc-macro dependency, then there currently is some ceremony
needed.
Place the proc-macro itself in a file like `auxiliary/my-proc-macro.rs`
with the following structure:

```rust,ignore
// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]

extern crate proc_macro;
use proc_macro::TokenStream;

#[proc_macro]
pub fn foo(input: TokenStream) -> TokenStream {
    "".parse().unwrap()
}
```

The `force-host` is needed because proc-macros are loaded in the host
compiler, and `no-prefer-dynamic` is needed to tell compiletest to not use
`prefer-dynamic` which is not compatible with proc-macros.
The `#![crate_type]` attribute is needed to specify the correct crate-type.

Then in your test, you can build with `aux-build`:

```rust,ignore
// aux-build: my-proc-macro.rs

extern crate my_proc_macro;

fn main() {
    my_proc_macro::foo!();
}
```


## Revisions

Revisions allow a single test file to be used for multiple tests.
This is done by adding a special header at the top of the file:

```rust,ignore
// revisions: foo bar baz
```

This will result in the test being compiled (and tested) three times,
once with `--cfg foo`, once with `--cfg bar`, and once with `--cfg
baz`.
You can therefore use `#[cfg(foo)]` etc within the test to tweak
each of these results.

You can also customize headers and expected error messages to a particular
revision. To do this, add `[foo]` (or `bar`, `baz`, etc) after the `//`
comment, like so:

```rust,ignore
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

<!-- date-check jul 2023 -->
Following is classes of tests that support revisions:
- UI
- assembly
- codegen
- debuginfo
- rustdoc UI tests
- incremental (these are special in that they inherently cannot be run in parallel)

## Compare modes

Compiletest can be run in different modes, called _compare modes_, which can
be used to compare the behavior of all tests with different compiler flags
enabled.
This can help highlight what differences might appear with certain flags, and
check for any problems that might arise.

To run the tests in a different mode, you need to pass the `--compare-mode`
CLI flag:

```bash
./x test tests/ui --compare-mode=chalk
```

The possible compare modes are:

* `polonius` — Runs with Polonius with `-Zpolonius`.
* `chalk` — Runs with Chalk with `-Zchalk`.
* `split-dwarf` — Runs with unpacked split-DWARF with `-Csplit-debuginfo=unpacked`.
* `split-dwarf-single` — Runs with packed split-DWARF with `-Csplit-debuginfo=packed`.

See [UI compare modes](ui.md#compare-modes) for more information about how UI
tests support different output for different modes.

In CI, compare modes are only used in one Linux builder, and only with the
following settings:

* `tests/debuginfo`: Uses `split-dwarf` mode.
  This helps ensure that none of the debuginfo tests are affected when
  enabling split-DWARF.

Note that compare modes are separate to [revisions](#revisions).
All revisions are tested when running `./x test tests/ui`, however
compare-modes must be manually run individually via the `--compare-mode` flag.
