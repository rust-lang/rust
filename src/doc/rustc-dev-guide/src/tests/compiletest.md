# Compiletest

<!-- toc -->

## Introduction

`compiletest` is the main test harness of the Rust test suite. It allows test
authors to organize large numbers of tests (the Rust compiler has many
thousands), efficient test execution (parallel execution is supported), and
allows the test author to configure behavior and expected results of both
individual and groups of tests.

> **Note for macOS users**
>
> For macOS users, `SIP` (System Integrity Protection) [may consistently check
> the compiled binary by sending network requests to Apple][zulip], so you may
> get a huge performance degradation when running tests.
>
> You can resolve it by tweaking the following settings: `Privacy & Security ->
> Developer Tools -> Add Terminal (Or VsCode, etc.)`.

[zulip]: https://rust-lang.zulipchat.com/#narrow/stream/182449-t-compiler.2Fhelp/topic/.E2.9C.94.20Is.20there.20any.20performance.20issue.20for.20MacOS.3F

`compiletest` may check test code for compile-time or run-time success/failure.

Tests are typically organized as a Rust source file with annotations in comments
before and/or within the test code. These comments serve to direct `compiletest`
on if or how to run the test, what behavior to expect, and more. See
[directives](directives.md) and the test suite documentation below for more details
on these annotations.

See the [Adding new tests](adding.md) and [Best practies](best-practices.md)
chapters for a tutorial on creating a new test and advice on writing a good
test, and the [Running tests](running.md) chapter on how to run the test suite.

Arguments can be passed to compiletest using `--test-args` or by placing them after `--`, e.g.
- `x test --test-args --force-rerun`
- `x test -- --force-rerun`

Additionally, bootstrap accepts several common arguments directly, e.g.

`x test --no-capture --force-rerun --run --pass`.

Compiletest itself tries to avoid running tests when the artifacts that are
involved (mainly the compiler) haven't changed. You can use `x test --test-args
--force-rerun` to rerun a test even when none of the inputs have changed.

## Test suites

All of the tests are in the [`tests`] directory. The tests are organized into
"suites", with each suite in a separate subdirectory. Each test suite behaves a
little differently, with different compiler behavior and different checks for
correctness. For example, the [`tests/incremental`] directory contains tests for
incremental compilation. The various suites are defined in
[`src/tools/compiletest/src/common.rs`] in the `pub enum Mode` declaration.

The following test suites are available, with links for more information:

### Compiler-specific test suites

| Test suite                                | Purpose                                                                                                             |
|-------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| [`ui`](ui.md)                             | Check the stdout/stderr snapshots from the compilation and/or running the resulting executable                      |
| `ui-fulldeps`                             | `ui` tests which require a linkable build of `rustc` (such as using `extern crate rustc_span;` or used as a plugin) |
| [`pretty`](#pretty-printer-tests)         | Check pretty printing                                                                                               |
| [`incremental`](#incremental-tests)       | Check incremental compilation behavior                                                                              |
| [`debuginfo`](#debuginfo-tests)           | Check debuginfo generation running debuggers                                                                        |
| [`codegen`](#codegen-tests)               | Check code generation                                                                                               |
| [`codegen-units`](#codegen-units-tests)   | Check codegen unit partitioning                                                                                     |
| [`assembly`](#assembly-tests)             | Check assembly output                                                                                               |
| [`mir-opt`](#mir-opt-tests)               | Check MIR generation and optimizations                                                                              |
| [`coverage`](#coverage-tests)             | Check coverage instrumentation                                                                                      |
| [`coverage-run-rustdoc`](#coverage-tests) | `coverage` tests that also run instrumented doctests                                                                |

### General purpose test suite

[`run-make`](#run-make-tests) are general purpose tests using Rust programs (or
Makefiles (legacy)).

### Rustdoc test suites

See [Rustdoc tests](../rustdoc.md#tests) for more details.

| Test suite       | Purpose                                                                  |
|------------------|--------------------------------------------------------------------------|
| `rustdoc`        | Check `rustdoc` generated files contain the expected documentation       |
| `rustdoc-gui`    | Check `rustdoc`'s GUI using a web browser                                |
| `rustdoc-js`     | Check `rustdoc` search is working as expected                            |
| `rustdoc-js-std` | Check rustdoc search is working as expected specifically on the std docs |
| `rustdoc-json`   | Check JSON output of `rustdoc`                                           |
| `rustdoc-ui`     | Check terminal output of `rustdoc`                                       |

[`tests`]: https://github.com/rust-lang/rust/blob/master/tests
[`src/tools/compiletest/src/common.rs`]: https://github.com/rust-lang/rust/tree/master/src/tools/compiletest/src/common.rs

### Pretty-printer tests

The tests in [`tests/pretty`] exercise the "pretty-printing" functionality of
`rustc`. The `-Z unpretty` CLI option for `rustc` causes it to translate the
input source into various different formats, such as the Rust source after macro
expansion.

The pretty-printer tests have several [directives](directives.md) described below.
These commands can significantly change the behavior of the test, but the
default behavior without any commands is to:

1. Run `rustc -Zunpretty=normal` on the source file.
2. Run `rustc -Zunpretty=normal` on the output of the previous step.
3. The output of the previous two steps should be the same.
4. Run `rustc -Zno-codegen` on the output to make sure that it can type check
   (this is similar to running `cargo check`).

If any of the commands above fail, then the test fails.

The directives for pretty-printing tests are:

- `pretty-mode` specifies the mode pretty-print tests should run in (that is,
  the argument to `-Zunpretty`). The default is `normal` if not specified.
- `pretty-compare-only` causes a pretty test to only compare the pretty-printed
  output (stopping after step 3 from above). It will not try to compile the
  expanded output to type check it. This is needed for a pretty-mode that does
  not expand to valid Rust, or for other situations where the expanded output
  cannot be compiled.
- `pp-exact` is used to ensure a pretty-print test results in specific output.
  If specified without a value, then it means the pretty-print output should
  match the original source. If specified with a value, as in `//@
  pp-exact:foo.pp`, it will ensure that the pretty-printed output matches the
  contents of the given file. Otherwise, if `pp-exact` is not specified, then
  the pretty-printed output will be pretty-printed one more time, and the output
  of the two pretty-printing rounds will be compared to ensure that the
  pretty-printed output converges to a steady state.

[`tests/pretty`]: https://github.com/rust-lang/rust/tree/master/tests/pretty

### Incremental tests

The tests in [`tests/incremental`] exercise incremental compilation. They use
[`revisions` directive](#revisions) to tell compiletest to run the compiler in a
series of steps.

Compiletest starts with an empty directory with the `-C incremental` flag, and
then runs the compiler for each revision, reusing the incremental results from
previous steps.

The revisions should start with:

* `rpass` — the test should compile and run successfully
* `rfail` — the test should compile successfully, but the executable should fail to run
* `cfail` — the test should fail to compile

To make the revisions unique, you should add a suffix like `rpass1` and
`rpass2`.

To simulate changing the source, compiletest also passes a `--cfg` flag with the
current revision name.

For example, this will run twice, simulating changing a function:

```rust,ignore
//@ revisions: rpass1 rpass2

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

`cfail` tests support the `forbid-output` directive to specify that a certain
substring must not appear anywhere in the compiler output. This can be useful to
ensure certain errors do not appear, but this can be fragile as error messages
change over time, and a test may no longer be checking the right thing but will
still pass.

`cfail` tests support the `should-ice` directive to specify that a test should
cause an Internal Compiler Error (ICE). This is a highly specialized directive
to check that the incremental cache continues to work after an ICE.

[`tests/incremental`]: https://github.com/rust-lang/rust/tree/master/tests/incremental


### Debuginfo tests

The tests in [`tests/debuginfo`] test debuginfo generation. They build a
program, launch a debugger, and issue commands to the debugger. A single test
can work with cdb, gdb, and lldb.

Most tests should have the `//@ compile-flags: -g` directive or something
similar to generate the appropriate debuginfo.

To set a breakpoint on a line, add a `// #break` comment on the line.

The debuginfo tests consist of a series of debugger commands along with
"check" lines which specify output that is expected from the debugger.

The commands are comments of the form `// $DEBUGGER-command:$COMMAND` where
`$DEBUGGER` is the debugger being used and `$COMMAND` is the debugger command
to execute.

The debugger values can be:

- `cdb`
- `gdb`
- `gdbg` — GDB without Rust support (versions older than 7.11)
- `gdbr` — GDB with Rust support
- `lldb`
- `lldbg` — LLDB without Rust support
- `lldbr` — LLDB with Rust support (this no longer exists)

The command to check the output are of the form `// $DEBUGGER-check:$OUTPUT`
where `$OUTPUT` is the output to expect.

For example, the following will build the test, start the debugger, set a
breakpoint, launch the program, inspect a value, and check what the debugger
prints:

```rust,ignore
//@ compile-flags: -g

//@ lldb-command: run
//@ lldb-command: print foo
//@ lldb-check: $0 = 123

fn main() {
    let foo = 123;
    b(); // #break
}

fn b() {}
```

The following [directives](directives.md) are available to disable a test based on
the debugger currently being used:

- `min-cdb-version: 10.0.18317.1001` — ignores the test if the version of cdb
  is below the given version
- `min-gdb-version: 8.2` — ignores the test if the version of gdb is below the
  given version
- `ignore-gdb-version: 9.2` — ignores the test if the version of gdb is equal
  to the given version
- `ignore-gdb-version: 7.11.90 - 8.0.9` — ignores the test if the version of
  gdb is in a range (inclusive)
- `min-lldb-version: 310` — ignores the test if the version of lldb is below
  the given version
- `rust-lldb` — ignores the test if lldb is not contain the Rust plugin. NOTE:
  The "Rust" version of LLDB doesn't exist anymore, so this will always be
  ignored. This should probably be removed.

By passing the `--debugger` option to compiletest, you can specify a single debugger to run tests with.
For example, `./x test tests/debuginfo -- --debugger gdb` will only test GDB commands.

> **Note on running lldb debuginfo tests locally**
>
> If you want to run lldb debuginfo tests locally, then currently on Windows it
> is required that:
> 
> - You have Python 3.10 installed.
> - You have `python310.dll` available in your `PATH` env var. This is not
>   provided by the standard Python installer you obtain from `python.org`; you
>   need to add this to `PATH` manually.
> 
> Otherwise the lldb debuginfo tests can produce crashes in mysterious ways.

[`tests/debuginfo`]: https://github.com/rust-lang/rust/tree/master/tests/debuginfo

> **Note on acquiring `cdb.exe` on Windows 11**
>
> `cdb.exe` is acquired alongside a suitable "Windows 11 SDK" which is part of
> the "Desktop Development with C++" workload profile in a Visual Studio
> installer (e.g. Visual Studio 2022 installer).
>
> **HOWEVER** this is not sufficient by default alone. If you need `cdb.exe`,
> you must go to Installed Apps, find the newest "Windows Software Development
> Kit" (and yes, this can still say `Windows 10.0.22161.3233` even though the OS
> is called Windows 11). You must then click "Modify" -> "Change" and then
> selected "Debugging Tools for Windows" in order to acquire `cdb.exe`.

### Codegen tests

The tests in [`tests/codegen`] test LLVM code generation. They compile the test
with the `--emit=llvm-ir` flag to emit LLVM IR. They then run the LLVM
[FileCheck] tool. The test is annotated with various `// CHECK` comments to
check the generated code. See the [FileCheck] documentation for a tutorial and
more information.

See also the [assembly tests](#assembly-tests) for a similar set of tests.

If you need to work with `#![no_std]` cross-compiling tests, consult the
[`minicore` test auxiliary](./minicore.md) chapter.

[`tests/codegen`]: https://github.com/rust-lang/rust/tree/master/tests/codegen
[FileCheck]: https://llvm.org/docs/CommandGuide/FileCheck.html


### Assembly tests

The tests in [`tests/assembly`] test LLVM assembly output. They compile the test
with the `--emit=asm` flag to emit a `.s` file with the assembly output. They
then run the LLVM [FileCheck] tool.

Each test should be annotated with the `//@ assembly-output:` directive with a
value of either `emit-asm` or `ptx-linker` to indicate the type of assembly
output.

Then, they should be annotated with various `// CHECK` comments to check the
assembly output. See the [FileCheck] documentation for a tutorial and more
information.

See also the [codegen tests](#codegen-tests) for a similar set of tests.

If you need to work with `#![no_std]` cross-compiling tests, consult the
[`minicore` test auxiliary](./minicore.md) chapter.

[`tests/assembly`]: https://github.com/rust-lang/rust/tree/master/tests/assembly


### Codegen-units tests

The tests in [`tests/codegen-units`] test the
[monomorphization](../backend/monomorph.md) collector and CGU partitioning.

These tests work by running `rustc` with a flag to print the result of the
monomorphization collection pass, and then special annotations in the file are
used to compare against that.

Each test should be annotated with the `//@
compile-flags:-Zprint-mono-items=VAL` directive with the appropriate `VAL` to
instruct `rustc` to print the monomorphization information.

Then, the test should be annotated with comments of the form `//~ MONO_ITEM
name` where `name` is the monomorphized string printed by rustc like `fn <u32 as
Trait>::foo`.

To check for CGU partitioning, a comment of the form `//~ MONO_ITEM name @@ cgu`
where `cgu` is a space separated list of the CGU names and the linkage
information in brackets. For example: `//~ MONO_ITEM static function::FOO @@
statics[Internal]`

[`tests/codegen-units`]: https://github.com/rust-lang/rust/tree/master/tests/codegen-units


### Mir-opt tests

The tests in [`tests/mir-opt`] check parts of the generated MIR to make sure it
is generated correctly and is doing the expected optimizations. Check out the
[MIR Optimizations](../mir/optimizations.md) chapter for more.

Compiletest will build the test with several flags to dump the MIR output and
set a baseline for optimizations:

* `-Copt-level=1`
* `-Zdump-mir=all`
* `-Zmir-opt-level=4`
* `-Zvalidate-mir`
* `-Zdump-mir-exclude-pass-number`

The test should be annotated with `// EMIT_MIR` comments that specify files that
will contain the expected MIR output. You can use `x test --bless` to create the
initial expected files.

There are several forms the `EMIT_MIR` comment can take:

- `// EMIT_MIR $MIR_PATH.mir` — This will check that the given filename matches
  the exact output from the MIR dump. For example,
  `my_test.main.SimplifyCfg-elaborate-drops.after.mir` will load that file from
  the test directory, and compare it against the dump from rustc.

  Checking the "after" file (which is after optimization) is useful if you are
  interested in the final state after an optimization. Some rare cases may want
  to use the "before" file for completeness.

- `// EMIT_MIR $MIR_PATH.diff` — where `$MIR_PATH` is the filename of the MIR
  dump, such as `my_test_name.my_function.EarlyOtherwiseBranch`. Compiletest
  will diff the `.before.mir` and `.after.mir` files, and compare the diff
  output to the expected `.diff` file from the `EMIT_MIR` comment.

  This is useful if you want to see how an optimization changes the MIR.

- `// EMIT_MIR $MIR_PATH.dot` — When using specific flags that dump additional
  MIR data (e.g. `-Z dump-mir-graphviz` to produce `.dot` files), this will
  check that the output matches the given file.

By default 32 bit and 64 bit targets use the same dump files, which can be
problematic in the presence of pointers in constants or other bit width
dependent things. In that case you can add `// EMIT_MIR_FOR_EACH_BIT_WIDTH` to
your test, causing separate files to be generated for 32bit and 64bit systems.

[`tests/mir-opt`]: https://github.com/rust-lang/rust/tree/master/tests/mir-opt


### `run-make` tests

> **Note on phasing out `Makefile`s**
> 
> We are planning to migrate all existing Makefile-based `run-make` tests
> to Rust programs. You should not be adding new Makefile-based `run-make`
> tests.
>
> See <https://github.com/rust-lang/rust/issues/121876>.

The tests in [`tests/run-make`] are general-purpose tests using Rust *recipes*,
which are small programs (`rmake.rs`) allowing arbitrary Rust code such as
`rustc` invocations, and is supported by a [`run_make_support`] library. Using
Rust recipes provide the ultimate in flexibility.

`run-make` tests should be used if no other test suites better suit your needs.

#### Using Rust recipes

Each test should be in a separate directory with a `rmake.rs` Rust program,
called the *recipe*. A recipe will be compiled and executed by compiletest with
the `run_make_support` library linked in.

If you need new utilities or functionality, consider extending and improving the
[`run_make_support`] library.

Compiletest directives like `//@ only-<target>` or `//@ ignore-<target>` are
supported in `rmake.rs`, like in UI tests. However, revisions or building
auxiliary via directives are not currently supported.

Two `run-make` tests are ported over to Rust recipes as examples:

- <https://github.com/rust-lang/rust/tree/master/tests/run-make/CURRENT_RUSTC_VERSION>
- <https://github.com/rust-lang/rust/tree/master/tests/run-make/a-b-a-linker-guard>

#### Quickly check if `rmake.rs` tests can be compiled

You can quickly check if `rmake.rs` tests can be compiled without having to
build stage1 rustc by forcing `rmake.rs` to be compiled with the stage0
compiler:

```bash
$ COMPILETEST_FORCE_STAGE0=1 x test --stage 0 tests/run-make/<test-name>
```

Of course, some tests will not successfully *run* in this way.

#### Using rust-analyzer with `rmake.rs`

Like other test programs, the `rmake.rs` scripts used by run-make tests do not
have rust-analyzer integration by default.

To work around this when working on a particular test, temporarily create a
`Cargo.toml` file in the test's directory
(e.g. `tests/run-make/sysroot-crates-are-unstable/Cargo.toml`)
with these contents:

<div class="warning">
Be careful not to add this `Cargo.toml` or its `Cargo.lock` to your actual PR!
</div>

```toml
# Convince cargo that this isn't part of an enclosing workspace.
[workspace]

[package]
name = "rmake"
version = "0.1.0"
edition = "2021"

[dependencies]
run_make_support = { path = "../../../src/tools/run-make-support" }

[[bin]]
name = "rmake"
path = "rmake.rs"
```

Then add a corresponding entry to `"rust-analyzer.linkedProjects"`
(e.g. in `.vscode/settings.json`):

```json
"rust-analyzer.linkedProjects": [
  "tests/run-make/sysroot-crates-are-unstable/Cargo.toml"
],
```

#### Using Makefiles (legacy)

<div class="warning">
You should avoid writing new Makefile-based `run-make` tests.
</div>

Each test should be in a separate directory with a `Makefile` indicating the
commands to run.

There is a [`tools.mk`] Makefile which you can include which provides a bunch of
utilities to make it easier to run commands and compare outputs. Take a look at
some of the other tests for some examples on how to get started.

[`tools.mk`]: https://github.com/rust-lang/rust/blob/master/tests/run-make/tools.mk
[`tests/run-make`]: https://github.com/rust-lang/rust/tree/master/tests/run-make
[`run_make_support`]: https://github.com/rust-lang/rust/tree/master/src/tools/run-make-support

### Coverage tests

The tests in [`tests/coverage`] are shared by multiple test modes that test
coverage instrumentation in different ways. Running the `coverage` test suite
will automatically run each test in all of the different coverage modes.

Each mode also has an alias to run the coverage tests in just that mode:

```bash
./x test coverage # runs all of tests/coverage in all coverage modes
./x test tests/coverage # same as above

./x test tests/coverage/if.rs # runs the specified test in all coverage modes

./x test coverage-map # runs all of tests/coverage in "coverage-map" mode only
./x test coverage-run # runs all of tests/coverage in "coverage-run" mode only

./x test coverage-map -- tests/coverage/if.rs # runs the specified test in "coverage-map" mode only
```

If a particular test should not be run in one of the coverage test modes for
some reason, use the `//@ ignore-coverage-map` or `//@ ignore-coverage-run`
directives.

#### `coverage-map` suite

In `coverage-map` mode, these tests verify the mappings between source code
regions and coverage counters that are emitted by LLVM. They compile the test
with `--emit=llvm-ir`, then use a custom tool ([`src/tools/coverage-dump`]) to
extract and pretty-print the coverage mappings embedded in the IR. These tests
don't require the profiler runtime, so they run in PR CI jobs and are easy to
run/bless locally.

These coverage map tests can be sensitive to changes in MIR lowering or MIR
optimizations, producing mappings that are different but produce identical
coverage reports.

As a rule of thumb, any PR that doesn't change coverage-specific code should
**feel free to re-bless** the `coverage-map` tests as necessary, without
worrying about the actual changes, as long as the `coverage-run` tests still
pass.

#### `coverage-run` suite

In `coverage-run` mode, these tests perform an end-to-end test of coverage
reporting. They compile a test program with coverage instrumentation, run that
program to produce raw coverage data, and then use LLVM tools to process that
data into a human-readable code coverage report.

Instrumented binaries need to be linked against the LLVM profiler runtime, so
`coverage-run` tests are **automatically skipped** unless the profiler runtime
is enabled in `bootstrap.toml`:

```toml
# bootstrap.toml
[build]
profiler = true
```

This also means that they typically don't run in PR CI jobs, though they do run
as part of the full set of CI jobs used for merging.

#### `coverage-run-rustdoc` suite

The tests in [`tests/coverage-run-rustdoc`] also run instrumented doctests and
include them in the coverage report. This avoids having to build rustdoc when
only running the main `coverage` suite.

[`tests/coverage`]: https://github.com/rust-lang/rust/tree/master/tests/coverage
[`src/tools/coverage-dump`]: https://github.com/rust-lang/rust/tree/master/src/tools/coverage-dump
[`tests/coverage-run-rustdoc`]: https://github.com/rust-lang/rust/tree/master/tests/coverage-run-rustdoc

### Crashes tests

[`tests/crashes`] serve as a collection of tests that are expected to cause the
compiler to ICE, panic or crash in some other way, so that accidental fixes are
tracked. This was formally done at <https://github.com/rust-lang/glacier> but
doing it inside the rust-lang/rust testsuite is more convenient.

It is imperative that a test in the suite causes rustc to ICE, panic or crash
crash in some other way. A test will "pass" if rustc exits with an exit status
other than 1 or 0.

If you want to see verbose stdout/stderr, you need to set
`COMPILETEST_VERBOSE_CRASHES=1`, e.g.

```bash
$ COMPILETEST_VERBOSE_CRASHES=1 ./x test tests/crashes/999999.rs --stage 1
```

When adding crashes from <https://github.com/rust-lang/rust/issues>, the issue
number should be noted in the file name (`12345.rs` should suffice) and also
inside the file include a `//@ known-bug: #4321` directive.

If you happen to fix one of the crashes, please move it to a fitting
subdirectory in `tests/ui` and give it a meaningful name. Please add a doc
comment at the top of the file explaining why this test exists, even better if
you can briefly explain how the example causes rustc to crash previously and
what was done to prevent rustc to ICE/panic/crash.

Adding

```text
Fixes #NNNNN
Fixes #MMMMM
```

to the description of your pull request will ensure the corresponding tickets be closed
automatically upon merge.

Make sure that your fix actually fixes the root cause of the issue and not just
a subset first. The issue numbers can be found in the file name or the `//@
known-bug` directive inside the test file.

[`tests/crashes`]: https://github.com/rust-lang/rust/tree/master/tests/crashes

## Building auxiliary crates

It is common that some tests require additional auxiliary crates to be compiled.
There are multiple [directives](directives.md) to assist with that:

- `aux-build`
- `aux-crate`
- `aux-bin`
- `aux-codegen-backend`
- `proc-macro`

`aux-build` will build a separate crate from the named source file. The source
file should be in a directory called `auxiliary` beside the test file.

```rust,ignore
//@ aux-build: my-helper.rs

extern crate my_helper;
// ... You can use my_helper.
```

The aux crate will be built as a dylib if possible (unless on a platform that
does not support them, or the `no-prefer-dynamic` header is specified in the aux
file). The `-L` flag is used to find the extern crates.

`aux-crate` is very similar to `aux-build`. However, it uses the `--extern` flag
to link to the extern crate to make the crate be available as an extern prelude.
That allows you to specify the additional syntax of the `--extern` flag, such as
renaming a dependency. For example, `// aux-crate:foo=bar.rs` will compile
`auxiliary/bar.rs` and make it available under then name `foo` within the test.
This is similar to how Cargo does dependency renaming.

`aux-bin` is similar to `aux-build` but will build a binary instead of a
library. The binary will be available in `auxiliary/bin` relative to the working
directory of the test.

`aux-codegen-backend` is similar to `aux-build`, but will then pass the compiled
dylib to `-Zcodegen-backend` when building the main file. This will only work
for tests in `tests/ui-fulldeps`, since it requires the use of compiler crates.

### Auxiliary proc-macro

If you want a proc-macro dependency, then you can use the `proc-macro`
directive. This directive behaves just like `aux-build`, i.e. that you should
place the proc-macro test auxiliary file under a `auxiliary` folder under the
same parent folder as the main test file. However, it also has four additional
preset behavior compared to `aux-build` for the proc-macro test auxiliary:

1. The aux test file is built with `--crate-type=proc-macro`.
2. The aux test file is built without `-C prefer-dynamic`, i.e. it will not try
   to produce a dylib for the aux crate.
3. The aux crate is made available to the test file via extern prelude with
   `--extern <aux_crate_name>`. Note that since UI tests default to edition
   2015, you still need to specify `extern <aux_crate_name>` unless the main
   test file is using an edition that is 2018 or newer if you want to use the
   aux crate name in a `use` import.
4. The `proc_macro` crate is made available as an extern prelude module. Same
   edition 2015 vs newer edition distinction for `extern proc_macro;` applies.

For example, you might have a test `tests/ui/cat/meow.rs` and proc-macro
auxiliary `tests/ui/cat/auxiliary/whiskers.rs`:

```text
tests/ui/cat/
    meow.rs                 # main test file
    auxiliary/whiskers.rs   # auxiliary
```

```rs
// tests/ui/cat/meow.rs

//@ proc-macro: whiskers.rs

extern crate whiskers; // needed as ui test defaults to edition 2015

fn main() {
  whiskers::identity!();
}
```

```rs
// tests/ui/cat/auxiliary/whiskers.rs

extern crate proc_macro;
use proc_macro::*;

#[proc_macro]
pub fn identity(ts: TokenStream) -> TokenStream {
    ts
}
```

> **Note**: The `proc-macro` header currently does not work with the
> `build-aux-doc` header for rustdoc tests. In that case, you will need to use
> the `aux-build` header, and use `#![crate_type="proc_macro"]`, and `//@
> force-host` and `//@ no-prefer-dynamic` headers in the proc-macro.

## Revisions

Revisions allow a single test file to be used for multiple tests. This is done
by adding a special directive at the top of the file:

```rust,ignore
//@ revisions: foo bar baz
```

This will result in the test being compiled (and tested) three times, once with
`--cfg foo`, once with `--cfg bar`, and once with `--cfg baz`. You can therefore
use `#[cfg(foo)]` etc within the test to tweak each of these results.

You can also customize directives and expected error messages to a particular
revision. To do this, add `[revision-name]` after the `//@` for directives, and
after `//` for UI error annotations, like so:

```rust,ignore
// A flag to pass in only for cfg `foo`:
//@[foo]compile-flags: -Z verbose-internals

#[cfg(foo)]
fn test_foo() {
    let x: usize = 32_u32; //[foo]~ ERROR mismatched types
}
```

Multiple revisions can be specified in a comma-separated list, such as
`//[foo,bar,baz]~^`.

In test suites that use the LLVM [FileCheck] tool, the current revision name is
also registered as an additional prefix for FileCheck directives:

```rust,ignore
//@ revisions: NORMAL COVERAGE
//@[COVERAGE] compile-flags: -Cinstrument-coverage
//@[COVERAGE] needs-profiler-runtime

// COVERAGE:   @__llvm_coverage_mapping
// NORMAL-NOT: @__llvm_coverage_mapping

// CHECK: main
fn main() {}
```

Note that not all directives have meaning when customized to a revision. For
example, the `ignore-test` directives (and all "ignore" directives) currently
only apply to the test as a whole, not to particular revisions. The only
directives that are intended to really work when customized to a revision are
error patterns and compiler flags.

<!-- date-check jul 2023 -->
The following test suites support revisions:

- ui
- assembly
- codegen
- coverage
- debuginfo
- rustdoc UI tests
- incremental (these are special in that they inherently cannot be run in
  parallel)

### Ignoring unused revision names

Normally, revision names mentioned in other directives and error annotations
must correspond to an actual revision declared in a `revisions` directive. This is
enforced by an `./x test tidy` check.

If a revision name needs to be temporarily removed from the revision list for
some reason, the above check can be suppressed by adding the revision name to an
`//@ unused-revision-names:` header instead.

Specifying an unused name of `*` (i.e. `//@ unused-revision-names: *`) will
permit any unused revision name to be mentioned.

## Compare modes

Compiletest can be run in different modes, called _compare modes_, which can be
used to compare the behavior of all tests with different compiler flags enabled.
This can help highlight what differences might appear with certain flags, and
check for any problems that might arise.

To run the tests in a different mode, you need to pass the `--compare-mode` CLI
flag:

```bash
./x test tests/ui --compare-mode=chalk
```

The possible compare modes are:

- `polonius` — Runs with Polonius with `-Zpolonius`.
- `chalk` — Runs with Chalk with `-Zchalk`.
- `split-dwarf` — Runs with unpacked split-DWARF with
  `-Csplit-debuginfo=unpacked`.
- `split-dwarf-single` — Runs with packed split-DWARF with
  `-Csplit-debuginfo=packed`.

See [UI compare modes](ui.md#compare-modes) for more information about how UI
tests support different output for different modes.

In CI, compare modes are only used in one Linux builder, and only with the
following settings:

- `tests/debuginfo`: Uses `split-dwarf` mode. This helps ensure that none of the
  debuginfo tests are affected when enabling split-DWARF.

Note that compare modes are separate to [revisions](#revisions). All revisions
are tested when running `./x test tests/ui`, however compare-modes must be
manually run individually via the `--compare-mode` flag.
