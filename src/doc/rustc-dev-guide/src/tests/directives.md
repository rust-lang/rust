# Compiletest directives

<!-- toc -->

<!--
FIXME(jieyouxu) completely revise this chapter.
-->

Directives are special comments that tell compiletest how to build and interpret a test.
They may also appear in `rmake.rs` [run-make tests](compiletest.md#run-make-tests).

They are normally put after the short comment that explains the point of this
test. Compiletest test suites use `//@` to signal that a comment is a directive.
For example, this test uses the `//@ compile-flags` command to specify a custom
flag to give to rustc when the test is compiled:

```rust,ignore
// Test the behavior of `0 - 1` when overflow checks are disabled.

//@ compile-flags: -C overflow-checks=off

fn main() {
    let x = 0 - 1;
    ...
}
```

Directives can be standalone (like `//@ run-pass`) or take a value (like `//@
compile-flags: -C overflow-checks=off`).

Directives are written one directive per line: you cannot write multiple
directives on the same line. For example, if you write `//@ only-x86
only-windows` then `only-windows` is interpreted as a comment, not a separate
directive.

## Listing of compiletest directives

The following is a list of compiletest directives. Directives are linked to
sections that describe the command in more detail if available. This list may
not be exhaustive. Directives can generally be found by browsing the
`TestProps` structure found in [`header.rs`] from the compiletest source.

[`header.rs`]: https://github.com/rust-lang/rust/tree/master/src/tools/compiletest/src/header.rs

### Assembly

<!-- date-check: Oct 2024 -->

| Directive         | Explanation                   | Supported test suites | Possible values                        |
|-------------------|-------------------------------|-----------------------|----------------------------------------|
| `assembly-output` | Assembly output kind to check | `assembly`            | `emit-asm`, `bpf-linker`, `ptx-linker` |

### Auxiliary builds

| Directive             | Explanation                                                                                           | Supported test suites | Possible values                               |
|-----------------------|-------------------------------------------------------------------------------------------------------|-----------------------|-----------------------------------------------|
| `aux-bin`             | Build a aux binary, made available in `auxiliary/bin` relative to test directory                      | All except `run-make` | Path to auxiliary `.rs` file                  |
| `aux-build`           | Build a separate crate from the named source file                                                     | All except `run-make` | Path to auxiliary `.rs` file                  |
| `aux-crate`           | Like `aux-build` but makes available as extern prelude                                                | All except `run-make` | `<extern_prelude_name>=<path/to/aux/file.rs>` |
| `aux-codegen-backend` | Similar to `aux-build` but pass the compiled dylib to `-Zcodegen-backend` when building the main file | `ui-fulldeps`         | Path to codegen backend file                  |
| `proc-macro`          | Similar to `aux-build`, but for aux forces host and don't use `-Cprefer-dynamic`[^pm].                | All except `run-make` | Path to auxiliary proc-macro `.rs` file       |
| `build-aux-docs`      | Build docs for auxiliaries as well                                                                    | All except `run-make` | N/A                                           |

[^pm]: please see the Auxiliary proc-macro section in the
    [compiletest](./compiletest.md) chapter for specifics.

### Controlling outcome expectations

See [Controlling pass/fail
expectations](ui.md#controlling-passfail-expectations).

| Directive                   | Explanation                                 | Supported test suites                     | Possible values |
|-----------------------------|---------------------------------------------|-------------------------------------------|-----------------|
| `check-pass`                | Building (no codegen) should pass           | `ui`, `crashes`, `incremental`            | N/A             |
| `check-fail`                | Building (no codegen) should fail           | `ui`, `crashes`                           | N/A             |
| `build-pass`                | Building should pass                        | `ui`, `crashes`, `codegen`, `incremental` | N/A             |
| `build-fail`                | Building should fail                        | `ui`, `crashes`                           | N/A             |
| `run-pass`                  | Running the test binary should pass         | `ui`, `crashes`, `incremental`            | N/A             |
| `run-fail`                  | Running the test binary should fail         | `ui`, `crashes`                           | N/A             |
| `ignore-pass`               | Ignore `--pass` flag                        | `ui`, `crashes`, `codegen`, `incremental` | N/A             |
| `dont-check-failure-status` | Don't check exact failure status (i.e. `1`) | `ui`, `incremental`                       | N/A             |
| `failure-status`            | Check                                       | `ui`, `crashes`                           | Any `u16`       |
| `should-ice`                | Check failure status is `101`               | `coverage`, `incremental`                 | N/A             |
| `should-fail`               | Compiletest self-test                       | All                                       | N/A             |

### Controlling output snapshots and normalizations

See [Normalization](ui.md#normalization), [Output
comparison](ui.md#output-comparison) and [Rustfix tests](ui.md#rustfix-tests)
for more details.

| Directive                         | Explanation                                                                                                              | Supported test suites                        | Possible values                                                                         |
|-----------------------------------|--------------------------------------------------------------------------------------------------------------------------|----------------------------------------------|-----------------------------------------------------------------------------------------|
| `check-run-results`               | Check run test binary `run-{pass,fail}` output snapshot                                                                  | `ui`, `crashes`, `incremental` if `run-pass` | N/A                                                                                     |
| `error-pattern`                   | Check that output contains a specific string                                                                             | `ui`, `crashes`, `incremental` if `run-pass` | String                                                                                  |
| `regex-error-pattern`             | Check that output contains a regex pattern                                                                               | `ui`, `crashes`, `incremental` if `run-pass` | Regex                                                                                   |
| `check-stdout`                    | Check `stdout` against `error-pattern`s from running test binary[^check_stdout]                                          | `ui`, `crashes`, `incremental`               | N/A                                                                                     |
| `normalize-stderr-32bit`          | Normalize actual stderr (for 32-bit platforms) with a rule `"<raw>" -> "<normalized>"` before comparing against snapshot | `ui`, `incremental`                          | `"<RAW>" -> "<NORMALIZED>"`, `<RAW>`/`<NORMALIZED>` is regex capture and replace syntax |
| `normalize-stderr-64bit`          | Normalize actual stderr (for 64-bit platforms) with a rule `"<raw>" -> "<normalized>"` before comparing against snapshot | `ui`, `incremental`                          | `"<RAW>" -> "<NORMALIZED>"`, `<RAW>`/`<NORMALIZED>` is regex capture and replace syntax |
| `normalize-stderr`                | Normalize actual stderr with a rule `"<raw>" -> "<normalized>"` before comparing against snapshot                        | `ui`, `incremental`                          | `"<RAW>" -> "<NORMALIZED>"`, `<RAW>`/`<NORMALIZED>` is regex capture and replace syntax |
| `normalize-stdout`                | Normalize actual stdout with a rule `"<raw>" -> "<normalized>"` before comparing against snapshot                        | `ui`, `incremental`                          | `"<RAW>" -> "<NORMALIZED>"`, `<RAW>`/`<NORMALIZED>` is regex capture and replace syntax |
| `dont-check-compiler-stderr`      | Don't check actual compiler stderr vs stderr snapshot                                                                    | `ui`                                         | N/A                                                                                     |
| `dont-check-compiler-stdout`      | Don't check actual compiler stdout vs stdout snapshot                                                                    | `ui`                                         | N/A                                                                                     |
| `dont-require-annotations`        | Don't require line annotations for the given diagnostic kind (`//~ KIND`) to be exhaustive                               | `ui`, `incremental`                          | `ERROR`, `WARN`, `NOTE`, `HELP`, `SUGGESTION`                                           |
| `run-rustfix`                     | Apply all suggestions via `rustfix`, snapshot fixed output, and check fixed output builds                                | `ui`                                         | N/A                                                                                     |
| `rustfix-only-machine-applicable` | `run-rustfix` but only machine-applicable suggestions                                                                    | `ui`                                         | N/A                                                                                     |
| `exec-env`                        | Env var to set when executing a test                                                                                     | `ui`, `crashes`                              | `<KEY>=<VALUE>`                                                                         |
| `unset-exec-env`                  | Env var to unset when executing a test                                                                                   | `ui`, `crashes`                              | Any env var name                                                                        |
| `stderr-per-bitwidth`             | Generate a stderr snapshot for each bitwidth                                                                             | `ui`                                         | N/A                                                                                     |
| `forbid-output`                   | A pattern which must not appear in stderr/`cfail` output                                                                 | `ui`, `incremental`                          | Regex pattern                                                                           |
| `run-flags`                       | Flags passed to the test executable                                                                                      | `ui`                                         | Arbitrary flags                                                                         |
| `known-bug`                       | No error annotation needed due to known bug                                                                              | `ui`, `crashes`, `incremental`               | Issue number `#123456`                                                                  |

[^check_stdout]: presently <!-- date-check: Oct 2024 --> this has a weird quirk
    where the test binary's stdout and stderr gets concatenated and then
    `error-pattern`s are matched on this combined output, which is ??? slightly
    questionable to say the least.

### Controlling when tests are run

These directives are used to ignore the test in some situations, which
means the test won't be compiled or run.

* `ignore-X` where `X` is a target detail or other criteria on which to ignore the test (see below)
* `only-X` is like `ignore-X`, but will *only* run the test on that target or
  stage
* `ignore-auxiliary` is intended for files that *participate* in one or more other
  main test files but that `compiletest` should not try to build the file itself.
  Please backlink to which main test is actually using the auxiliary file.
* `ignore-test` always ignores the test. This can be used to temporarily disable
  a test if it is currently not working, but you want to keep it in tree to
  re-enable it later.

Some examples of `X` in `ignore-X` or `only-X`:

- A full target triple: `aarch64-apple-ios`
- Architecture: `aarch64`, `arm`, `mips`, `wasm32`, `x86_64`, `x86`,
  ...
- OS: `android`, `emscripten`, `freebsd`, `ios`, `linux`, `macos`, `windows`,
  ...
- Environment (fourth word of the target triple): `gnu`, `msvc`, `musl`
- WASM: `wasm32-bare` matches `wasm32-unknown-unknown`. `emscripten` also
  matches that target as well as the emscripten targets.
- Pointer width: `32bit`, `64bit`
- Endianness: `endian-big`
- Stage: `stage1`, `stage2`
- Binary format: `elf`
- Channel: `stable`, `beta`
- When cross compiling: `cross-compile`
- When [remote testing] is used: `remote`
- When particular debuggers are being tested: `cdb`, `gdb`, `lldb`
- When particular debugger versions are matched: `ignore-gdb-version`
- Specific [compare modes]: `compare-mode-polonius`, `compare-mode-chalk`,
  `compare-mode-split-dwarf`, `compare-mode-split-dwarf-single`
- The two different test modes used by coverage tests:
  `ignore-coverage-map`, `ignore-coverage-run`
- When testing a dist toolchain: `dist`
  - This needs to be enabled with `COMPILETEST_ENABLE_DIST_TESTS=1`
- The `rustc_abi` of the target: e.g. `rustc_abi-x86_64-sse2`

The following directives will check rustc build settings and target
settings:

- `needs-asm-support` — ignores if it is running on a target that doesn't have
  stable support for `asm!`
- `needs-profiler-runtime` — ignores the test if the profiler runtime was not
  enabled for the target
  (`build.profiler = true` in rustc's `bootstrap.toml`)
- `needs-sanitizer-support` — ignores if the sanitizer support was not enabled
  for the target (`sanitizers = true` in rustc's `bootstrap.toml`)
- `needs-sanitizer-{address,hwaddress,leak,memory,thread}` — ignores if the
  corresponding sanitizer is not enabled for the target (AddressSanitizer,
  hardware-assisted AddressSanitizer, LeakSanitizer, MemorySanitizer or
  ThreadSanitizer respectively)
- `needs-run-enabled` — ignores if it is a test that gets executed, and running
  has been disabled. Running tests can be disabled with the `x test --run=never`
  flag, or running on fuchsia.
- `needs-unwind` — ignores if the target does not support unwinding
- `needs-rust-lld` — ignores if the rust lld support is not enabled (`rust.lld =
  true` in `bootstrap.toml`)
- `needs-threads` — ignores if the target does not have threading support
- `needs-subprocess`  — ignores if the target does not have subprocess support
- `needs-symlink` — ignores if the target does not support symlinks. This can be
  the case on Windows if the developer did not enable privileged symlink
  permissions.
- `ignore-std-debug-assertions` — ignores if std was built with debug
  assertions.
- `needs-std-debug-assertions` — ignores if std was not built with debug
  assertions.
- `ignore-rustc-debug-assertions` — ignores if rustc was built with debug
  assertions.
- `needs-rustc-debug-assertions` — ignores if rustc was not built with debug
  assertions.
- `needs-target-has-atomic` — ignores if target does not have support for all
  specified atomic widths, e.g. the test with `//@ needs-target-has-atomic: 8,
  16, ptr` will only run if it supports the comma-separated list of atomic
  widths.
- `needs-dynamic-linking` — ignores if target does not support dynamic linking
  (which is orthogonal to it being unable to create `dylib` and `cdylib` crate types)
- `needs-crate-type` — ignores if target platform does not support one or more
  of the comma-delimited list of specified crate types. For example,
  `//@ needs-crate-type: cdylib, proc-macro` will cause the test to be ignored
  on `wasm32-unknown-unknown` target because the target does not support the
  `proc-macro` crate type.

The following directives will check LLVM support:

- `exact-llvm-major-version: 19` — ignores if the llvm major version does not
  match the specified llvm major version.
- `min-llvm-version: 13.0` — ignored if the LLVM version is less than the given
  value
- `min-system-llvm-version: 12.0` — ignored if using a system LLVM and its
  version is less than the given value
- `max-llvm-major-version: 19` — ignored if the LLVM major version is higher
  than the given major version
- `ignore-llvm-version: 9.0` — ignores a specific LLVM version
- `ignore-llvm-version: 7.0 - 9.9.9` — ignores LLVM versions in a range
  (inclusive)
- `needs-llvm-components: powerpc` — ignores if the specific LLVM component was
  not built. Note: The test will fail on CI (when
  `COMPILETEST_REQUIRE_ALL_LLVM_COMPONENTS` is set) if the component does not
  exist.
- `needs-forced-clang-based-tests` — test is ignored unless the environment
  variable `RUSTBUILD_FORCE_CLANG_BASED_TESTS` is set, which enables building
  clang alongside LLVM
  - This is only set in two CI jobs ([`x86_64-gnu-debug`] and
    [`aarch64-gnu-debug`]), which only runs a
    subset of `run-make` tests. Other tests with this directive will not
    run at all, which is usually not what you want.

See also [Debuginfo tests](compiletest.md#debuginfo-tests) for directives for
ignoring debuggers.

[remote testing]: running.md#running-tests-on-a-remote-machine
[compare modes]: ui.md#compare-modes
[`x86_64-gnu-debug`]: https://github.com/rust-lang/rust/blob/ab3dba92db355b8d97db915a2dca161a117e959c/src/ci/docker/host-x86_64/x86_64-gnu-debug/Dockerfile#L32
[`aarch64-gnu-debug`]: https://github.com/rust-lang/rust/blob/20c909ff9cdd88d33768a4ddb8952927a675b0ad/src/ci/docker/host-aarch64/aarch64-gnu-debug/Dockerfile#L32

### Affecting how tests are built

| Directive           | Explanation                                                                                  | Supported test suites     | Possible values                                                                            |
|---------------------|----------------------------------------------------------------------------------------------|---------------------------|--------------------------------------------------------------------------------------------|
| `compile-flags`     | Flags passed to `rustc` when building the test or aux file                                   | All except for `run-make` | Any valid `rustc` flags, e.g. `-Awarnings -Dfoo`. Cannot be `-Cincremental` or `--edition` |
| `edition`           | The edition used to build the test                                                           | All except for `run-make` | Any valid `--edition` value                                                                |
| `rustc-env`         | Env var to set when running `rustc`                                                          | All except for `run-make` | `<KEY>=<VALUE>`                                                                            |
| `unset-rustc-env`   | Env var to unset when running `rustc`                                                        | All except for `run-make` | Any env var name                                                                           |
| `incremental`       | Proper incremental support for tests outside of incremental test suite                       | `ui`, `crashes`           | N/A                                                                                        |
| `no-prefer-dynamic` | Don't use `-C prefer-dynamic`, don't build as a dylib via a `--crate-type=dylib` preset flag | `ui`, `crashes`           | N/A                                                                                        |

<div class="warning">

Tests (outside of `run-make`) that want to use incremental tests not in the
incremental test-suite must not pass `-C incremental` via `compile-flags`, and
must instead use the `//@ incremental` directive.

Consider writing the test as a proper incremental test instead.

</div>

### Rustdoc

| Directive   | Explanation                                                  | Supported test suites                    | Possible values           |
|-------------|--------------------------------------------------------------|------------------------------------------|---------------------------|
| `doc-flags` | Flags passed to `rustdoc` when building the test or aux file | `rustdoc`, `rustdoc-js`, `rustdoc-json`  | Any valid `rustdoc` flags |

<!--
**FIXME(rustdoc)**: what does `check-test-line-numbers-match` do?
Asked in
<https://rust-lang.zulipchat.com/#narrow/stream/266220-t-rustdoc/topic/What.20is.20the.20.60check-test-line-numbers-match.60.20directive.3F>.
-->

#### Test-suite-specific directives

The test suites [`rustdoc`][rustdoc-html-tests], [`rustdoc-js`/`rustdoc-js-std`][rustdoc-js-tests]
and [`rustdoc-json`][rustdoc-json-tests] each feature an additional set of directives whose basic
syntax resembles the one of compiletest directives but which are ultimately read and checked by
separate tools. For more information, please read their respective chapters as linked above.

[rustdoc-html-tests]: ../rustdoc-internals/rustdoc-test-suite.md
[rustdoc-js-tests]: ../rustdoc-internals/search.html#testing-the-search-engine
[rustdoc-json-tests]: ../rustdoc-internals/rustdoc-json-test-suite.md

### Pretty printing

See [Pretty-printer](compiletest.md#pretty-printer-tests).

#### Misc directives

- `no-auto-check-cfg` — disable auto check-cfg (only for `--check-cfg` tests)
- [`revisions`](compiletest.md#revisions) — compile multiple times
- [`unused-revision-names`](compiletest.md#ignoring-unused-revision-names) -
      suppress tidy checks for mentioning unknown revision names
-[`forbid-output`](compiletest.md#incremental-tests) — incremental cfail rejects
      output pattern
- [`should-ice`](compiletest.md#incremental-tests) — incremental cfail should
      ICE
- [`reference`] — an annotation linking to a rule in the reference

[`reference`]: https://github.com/rust-lang/reference/blob/master/docs/authoring.md#test-rule-annotations

### Tool-specific directives

The following directives affect how certain command-line tools are invoked, in
test suites that use those tools:

- `filecheck-flags` adds extra flags when running LLVM's `FileCheck` tool.
  - Used by [codegen tests](compiletest.md#codegen-tests),
  [assembly tests](compiletest.md#assembly-tests), and
  [MIR-opt tests](compiletest.md#mir-opt-tests).
- `llvm-cov-flags` adds extra flags when running LLVM's `llvm-cov` tool.
  - Used by [coverage tests](compiletest.md#coverage-tests) in `coverage-run` mode.


## Substitutions

Directive values support substituting a few variables which will be replaced
with their corresponding value. For example, if you need to pass a compiler flag
with a path to a specific file, something like the following could work:

```rust,ignore
//@ compile-flags: --remap-path-prefix={{src-base}}=/the/src
```

Where the sentinel `{{src-base}}` will be replaced with the appropriate path
described below:

- `{{cwd}}`: The directory where compiletest is run from. This may not be the
  root of the checkout, so you should avoid using it where possible.
  - Examples: `/path/to/rust`, `/path/to/build/root`
- `{{src-base}}`: The directory where the test is defined. This is equivalent to
  `$DIR` for [output normalization].
  - Example: `/path/to/rust/tests/ui/error-codes`
- `{{build-base}}`: The base directory where the test's output goes. This is
  equivalent to `$TEST_BUILD_DIR` for [output normalization].
  - Example: `/path/to/rust/build/x86_64-unknown-linux-gnu/test/ui`
- `{{rust-src-base}}`: The sysroot directory where libstd/libcore/... are
  located
- `{{sysroot-base}}`: Path of the sysroot directory used to build the test.
  - Mainly intended for `ui-fulldeps` tests that run the compiler via API.
- `{{target-linker}}`: Linker that would be passed to `-Clinker` for this test,
  or blank if no linker override is active.
  - Mainly intended for `ui-fulldeps` tests that run the compiler via API.
- `{{target}}`: The target the test is compiling for
  - Example: `x86_64-unknown-linux-gnu`

See
[`tests/ui/commandline-argfile.rs`](https://github.com/rust-lang/rust/blob/master/tests/ui/argfile/commandline-argfile.rs)
for an example of a test that uses this substitution.

[output normalization]: ui.md#normalization


## Adding a directive

One would add a new directive if there is a need to define some test property or
behavior on an individual, test-by-test basis. A directive property serves as
the directive's backing store (holds the command's current value) at runtime.

To add a new directive property:

1. Look for the `pub struct TestProps` declaration in
   [`src/tools/compiletest/src/header.rs`] and add the new public property to
   the end of the declaration.
2. Look for the `impl TestProps` implementation block immediately following the
   struct declaration and initialize the new property to its default value.

### Adding a new directive parser

When `compiletest` encounters a test file, it parses the file a line at a time
by calling every parser defined in the `Config` struct's implementation block,
also in [`src/tools/compiletest/src/header.rs`] (note that the `Config` struct's
declaration block is found in [`src/tools/compiletest/src/common.rs`]).
`TestProps`'s `load_from()` method will try passing the current line of text to
each parser, which, in turn typically checks to see if the line begins with a
particular commented (`//@`) directive such as `//@ must-compile-successfully`
or `//@ failure-status`. Whitespace after the comment marker is optional.

Parsers will override a given directive property's default value merely by being
specified in the test file as a directive or by having a parameter value
specified in the test file, depending on the directive.

Parsers defined in `impl Config` are typically named `parse_<directive-name>`
(note kebab-case `<directive-command>` transformed to snake-case
`<directive_command>`). `impl Config` also defines several 'low-level' parsers
which make it simple to parse common patterns like simple presence or not
(`parse_name_directive()`), `directive:parameter(s)`
(`parse_name_value_directive()`), optional parsing only if a particular `cfg`
attribute is defined (`has_cfg_prefix()`) and many more. The low-level parsers
are found near the end of the `impl Config` block; be sure to look through them
and their associated parsers immediately above to see how they are used to avoid
writing additional parsing code unnecessarily.

As a concrete example, here is the implementation for the
`parse_failure_status()` parser, in [`src/tools/compiletest/src/header.rs`]:

```diff
@@ -232,6 +232,7 @@ pub struct TestProps {
     // customized normalization rules
     pub normalize_stdout: Vec<(String, String)>,
     pub normalize_stderr: Vec<(String, String)>,
+    pub failure_status: i32,
 }

 impl TestProps {
@@ -260,6 +261,7 @@ impl TestProps {
             run_pass: false,
             normalize_stdout: vec![],
             normalize_stderr: vec![],
+            failure_status: 101,
         }
     }

@@ -383,6 +385,10 @@ impl TestProps {
             if let Some(rule) = config.parse_custom_normalization(ln, "normalize-stderr") {
                 self.normalize_stderr.push(rule);
             }
+
+            if let Some(code) = config.parse_failure_status(ln) {
+                self.failure_status = code;
+            }
         });

         for key in &["RUST_TEST_NOCAPTURE", "RUST_TEST_THREADS"] {
@@ -488,6 +494,13 @@ impl Config {
         self.parse_name_directive(line, "pretty-compare-only")
     }

+    fn parse_failure_status(&self, line: &str) -> Option<i32> {
+        match self.parse_name_value_directive(line, "failure-status") {
+            Some(code) => code.trim().parse::<i32>().ok(),
+            _ => None,
+        }
+    }
```

### Implementing the behavior change

When a test invokes a particular directive, it is expected that some behavior
will change as a result. What behavior, obviously, will depend on the purpose of
the directive. In the case of `failure-status`, the behavior that changes is
that `compiletest` expects the failure code defined by the directive invoked in
the test, rather than the default value.

Although specific to `failure-status` (as every directive will have a different
implementation in order to invoke behavior change) perhaps it is helpful to see
the behavior change implementation of one case, simply as an example. To
implement `failure-status`, the `check_correct_failure_status()` function found
in the `TestCx` implementation block, located in
[`src/tools/compiletest/src/runtest.rs`], was modified as per below:

```diff
@@ -295,11 +295,14 @@ impl<'test> TestCx<'test> {
     }

     fn check_correct_failure_status(&self, proc_res: &ProcRes) {
-        // The value the Rust runtime returns on failure
-        const RUST_ERR: i32 = 101;
-        if proc_res.status.code() != Some(RUST_ERR) {
+        let expected_status = Some(self.props.failure_status);
+        let received_status = proc_res.status.code();
+
+        if expected_status != received_status {
             self.fatal_proc_rec(
-                &format!("failure produced the wrong error: {}", proc_res.status),
+                &format!("Error: expected failure status ({:?}) but received status {:?}.",
+                         expected_status,
+                         received_status),
                 proc_res,
             );
         }
@@ -320,7 +323,6 @@ impl<'test> TestCx<'test> {
         );

         let proc_res = self.exec_compiled_test();
-
         if !proc_res.status.success() {
             self.fatal_proc_rec("test run failed!", &proc_res);
         }
@@ -499,7 +501,6 @@ impl<'test> TestCx<'test> {
                 expected,
                 actual
             );
-            panic!();
         }
     }
```

Note the use of `self.props.failure_status` to access the directive property. In
tests which do not specify the failure status directive,
`self.props.failure_status` will evaluate to the default value of 101 at the
time of this writing. But for a test which specifies a directive of, for
example, `//@ failure-status: 1`, `self.props.failure_status` will evaluate to
1, as `parse_failure_status()` will have overridden the `TestProps` default
value, for that test specifically.

[`src/tools/compiletest/src/header.rs`]: https://github.com/rust-lang/rust/tree/master/src/tools/compiletest/src/header.rs
[`src/tools/compiletest/src/common.rs`]: https://github.com/rust-lang/rust/tree/master/src/tools/compiletest/src/common.rs
[`src/tools/compiletest/src/runtest.rs`]: https://github.com/rust-lang/rust/tree/master/src/tools/compiletest/src/runtest.rs
