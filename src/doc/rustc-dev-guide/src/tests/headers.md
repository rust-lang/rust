# Test headers

<!-- toc -->

Header commands are special comments that tell compiletest how to build and
interpret a test.
They must appear before the Rust source in the test.
They may also appear in Makefiles for [run-make tests](compiletest.md#run-make-tests).

They are normally put after the short comment that explains the point of this test.
For example, this test uses the `// compile-flags` command to specify a custom
flag to give to rustc when the test is compiled:

```rust,ignore
// Test the behavior of `0 - 1` when overflow checks are disabled.

// compile-flags: -C overflow-checks=off

fn main() {
    let x = 0 - 1;
    ...
}
```

Header commands can be standalone (like `// run-pass`) or take a value (like
`// compile-flags: -C overflow-checks=off`).

## Header commands

The following is a list of header commands.
Commands are linked to sections that describe the command in more detail if available.
This list may not be exhaustive.
Header commands can generally be found by browsing the `TestProps` structure
found in [`header.rs`] from the compiletest source.

[`header.rs`]: https://github.com/rust-lang/rust/tree/master/src/tools/compiletest/src/header.rs

* [Controlling pass/fail expectations](ui.md#controlling-passfail-expectations)
    * `check-pass` — building (no codegen) should pass
    * `build-pass` — building should pass
    * `run-pass` — running the test should pass
    * `check-fail` — building (no codegen) should fail (the default if no header)
    * `build-fail` — building should fail
    * `run-fail` — running should fail
    * `ignore-pass` — ignores the `--pass` flag
    * `check-run-results` — checks run-pass/fail-pass output
* [UI](ui.md) headers
    * [`normalize-X`](ui.md#normalization) — normalize compiler output
    * [`run-rustfix`](ui.md#rustfix-tests) — checks diagnostic suggestions
    * [`rustfix-only-machine-applicable`](ui.md#rustfix-tests) — checks only
      machine applicable suggestions
    * [`stderr-per-bitwidth`](ui.md#output-comparison) — separate output per bit width
    * [`dont-check-compiler-stderr`](ui.md#output-comparison) — don't validate stderr
    * [`dont-check-compiler-stdout`](ui.md#output-comparison) — don't validate stdout
    * [`compare-output-lines-by-subset`](ui.md#output-comparison) — checks output by
      line subset
* [Building auxiliary crates](compiletest.md#building-auxiliary-crates)
    * `aux-build`
    * `aux-crate`
* [Pretty-printer](compiletest.md#pretty-printer-tests) headers
    * `pretty-compare-only`
    * `pretty-expanded`
    * `pretty-mode`
    * `pp-exact`
* [Ignoring tests](#ignoring-tests)
    * `ignore-X`
    * `only-X`
    * `needs-X`
    * `no-system-llvm`
    * `min-llvm-versionX`
    * `min-system-llvm-version`
    * `ignore-llvm-version`
* [Environment variable headers](#environment-variable-headers)
    * `rustc-env`
    * `exec-env`
    * `unset-exec-env`
    * `unset-rustc-env`
* [Miscellaneous headers](#miscellaneous-headers)
    * `compile-flags` — adds compiler flags
    * `run-flags` — adds flags to executable tests
    * `edition` — sets the edition
    * `failure-status` — expected exit code
    * `should-fail` — testing compiletest itself
    * `gate-test-X` — feature gate testing
    * [`error-pattern`](ui.md#error-pattern) — errors not on a line
    * `incremental` — incremental tests not in the incremental test-suite
    * `no-prefer-dynamic` — don't use `-C prefer-dynamic`, don't build as a dylib
    * `force-host` — build only for the host target
    * [`revisions`](compiletest.md#revisions) — compile multiple times
    * [`forbid-output`](compiletest.md#incremental-tests) — incremental cfail rejects output pattern
    * [`should-ice`](compiletest.md#incremental-tests) — incremental cfail should ICE
    * [`known-bug`](ui.md#known-bugs) — indicates that the test is
      for a known bug that has not yet been fixed
* [Assembly](compiletest.md#assembly-tests) headers
    * `assembly-output` — the type of assembly output to check


### Ignoring tests

These header commands are used to ignore the test in some situations,
which means the test won't be compiled or run.

* `ignore-X` where `X` is a target detail or stage will ignore the
  test accordingly (see below)
* `only-X` is like `ignore-X`, but will *only* run the test on that
  target or stage
* `ignore-test` always ignores the test.
  This can be used to temporarily disable a test if it is currently not working,
  but you want to keep it in tree to re-enable it later.

Some examples of `X` in `ignore-X` or `only-X`:

* A full target triple: `aarch64-apple-ios`
* Architecture: `aarch64`, `arm`, `asmjs`, `mips`, `wasm32`, `x86_64`,
  `x86`, ...
* OS: `android`, `emscripten`, `freebsd`, `ios`, `linux`, `macos`,
  `windows`, ...
* Environment (fourth word of the target triple): `gnu`, `msvc`,
  `musl`
* WASM: `wasm32-bare` matches `wasm32-unknown-unknown`.
  `emscripten` also matches that target as well as the emscripten targets.
* Pointer width: `32bit`, `64bit`
* Endianness: `endian-big`
* Stage: `stage0`, `stage1`, `stage2`
* Channel: `stable`, `beta`
* When cross compiling: `cross-compile`
* When [remote testing] is used: `remote`
* When debug-assertions are enabled: `debug`
* When particular debuggers are being tested: `cdb`, `gdb`, `lldb`
* Specific [compare modes]: `compare-mode-polonius`,
  `compare-mode-chalk`, `compare-mode-split-dwarf`,
  `compare-mode-split-dwarf-single`

The following header commands will check rustc build settings and target settings:

* `needs-asm-support` — ignores if it is running on a target that doesn't have
  stable support for `asm!`
* `needs-profiler-support` — ignores if profiler support was not enabled for
  the target (`profiler = true` in rustc's `config.toml`)
* `needs-sanitizer-support` — ignores if the sanitizer support was not enabled
  for the target (`sanitizers = true` in rustc's `config.toml`)
* `needs-sanitizer-{address,hwaddress,leak,memory,thread}` — ignores
  if the corresponding sanitizer is not enabled for the target
  (AddressSanitizer, hardware-assisted AddressSanitizer, LeakSanitizer,
  MemorySanitizer or ThreadSanitizer respectively)
* `needs-run-enabled` — ignores if it is a test that gets executed, and
  running has been disabled. Running tests can be disabled with the `x test
  --run=never` flag, or running on fuchsia.
* `needs-unwind` — ignores if the target does not support unwinding
* `needs-rust-lld` — ignores if the rust lld support is not enabled
  (`rust.lld = true` in `config.toml`)

The following header commands will check LLVM support:

* `no-system-llvm` — ignores if the system llvm is used
* `min-llvm-version: 13.0` — ignored if the LLVM version is less than the given value
* `min-system-llvm-version: 12.0` — ignored if using a system LLVM and its
  version is less than the given value
* `ignore-llvm-version: 9.0` — ignores a specific LLVM version
* `ignore-llvm-version: 7.0 - 9.9.9` — ignores LLVM versions in a range (inclusive)
* `needs-llvm-components: powerpc` — ignores if the specific LLVM component was not built.
  Note: The test will fail on CI if the component does not exist.
* `needs-matching-clang` — ignores if the version of clang does not match the
  LLVM version of rustc.
  These tests are always ignored unless a special environment variable is set
  (which is only done in one CI job).

See also [Debuginfo tests](compiletest.md#debuginfo-tests) for headers for
ignoring debuggers.

[remote testing]: running.md#running-tests-on-a-remote-machine
[compare modes]: ui.md#compare-modes

### Environment variable headers

The following headers affect environment variables.

* `rustc-env` is an environment variable to set when running `rustc` of the
  form `KEY=VALUE`.
* `exec-env` is an environment variable to set when executing a test of the
  form `KEY=VALUE`.
* `unset-exec-env` specifies an environment variable to unset when executing a
  test.
* `unset-rustc-env` specifies an environment variable to unset when running
  `rustc`.

### Miscellaneous headers

The following headers are generally available, and not specific to particular
test suites.

* `compile-flags` passes extra command-line args to the compiler,
  e.g. `// compile-flags: -g` which forces debuginfo to be enabled.
* `run-flags` passes extra args to the test if the test is to be executed.
* `edition` controls the edition the test should be compiled with
  (defaults to 2015). Example usage: `// edition:2018`.
* `failure-status` specifies the numeric exit code that should be expected for
  tests that expect an error.
  If this is not set, the default is 1.
* `should-fail` indicates that the test should fail; used for "meta
  testing", where we test the compiletest program itself to check that
  it will generate errors in appropriate scenarios. This header is
  ignored for pretty-printer tests.
* `gate-test-X` where `X` is a feature marks the test as "gate test"
  for feature X.
  Such tests are supposed to ensure that the compiler errors when usage of a
  gated feature is attempted without the proper `#![feature(X)]` tag.
  Each unstable lang feature is required to have a gate test.
  This header is actually checked by [tidy](intro.md#tidy), it is not checked
  by compiletest.
* `error-pattern` checks the diagnostics just like the `ERROR` annotation
  without specifying error line. This is useful when the error doesn't give
  any span. See [`error-pattern`](ui.md#error-pattern).
* `incremental` runs the test with the `-C incremental` flag and an empty
  incremental directory. This should be avoided when possible; you should use
  an *incremental mode* test instead. Incremental mode tests support running
  the compiler multiple times and verifying that it can load the generated
  incremental cache. This flag is for specialized circumstances, like checking
  the interaction of codegen unit partitioning with generating an incremental
  cache.
* `no-prefer-dynamic` will force an auxiliary crate to be built as an rlib
  instead of a dylib. When specified in a test, it will remove the use of `-C
  prefer-dynamic`. This can be useful in a variety of circumstances. For
  example, it can prevent a proc-macro from being built with the wrong crate
  type. Or if your test is specifically targeting behavior of other crate
  types, it can be used to prevent building with the wrong crate type.
* `force-host` will force the test to build for the host platform instead of
  the target. This is useful primarily for auxiliary proc-macros, which need
  to be loaded by the host compiler.


## Substitutions

Headers values support substituting a few variables which will be replaced
with their corresponding value.
For example, if you need to pass a compiler flag with a path to a specific
file, something like the following could work:

```rust,ignore
// compile-flags: --remap-path-prefix={{src-base}}=/the/src
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

See [`tests/ui/commandline-argfile.rs`](https://github.com/rust-lang/rust/blob/master/tests/ui/commandline-argfile.rs)
for an example of a test that uses this substitution.

[output normalization]: ui.md#normalization


## Adding a new header command

One would add a new header command if there is a need to define some test
property or behavior on an individual, test-by-test basis.
A header command property serves as the header command's backing store (holds
the command's current value) at runtime.

To add a new header command property:

  1. Look for the `pub struct TestProps` declaration in
     [`src/tools/compiletest/src/header.rs`] and add the new public property to
     the end of the declaration.
  2. Look for the `impl TestProps` implementation block immediately following
     the struct declaration and initialize the new property to its default
     value.

### Adding a new header command parser

When `compiletest` encounters a test file, it parses the file a line at a time
by calling every parser defined in the `Config` struct's implementation block,
also in [`src/tools/compiletest/src/header.rs`] (note that the `Config`
struct's declaration block is found in [`src/tools/compiletest/src/common.rs`]).
`TestProps`'s `load_from()` method will try passing the current line of text to
each parser, which, in turn typically checks to see if the line begins with a
particular commented (`//`) header command such as `// must-compile-successfully`
or `// failure-status`. Whitespace after the comment marker is optional.

Parsers will override a given header command property's default value merely by
being specified in the test file as a header command or by having a parameter
value specified in the test file, depending on the header command.

Parsers defined in `impl Config` are typically named `parse_<header_command>`
(note kebab-case `<header-command>` transformed to snake-case
`<header_command>`). `impl Config` also defines several 'low-level' parsers
which make it simple to parse common patterns like simple presence or not
(`parse_name_directive()`), header-command:parameter(s)
(`parse_name_value_directive()`), optional parsing only if a particular `cfg`
attribute is defined (`has_cfg_prefix()`) and many more. The low-level parsers
are found near the end of the `impl Config` block; be sure to look through them
and their associated parsers immediately above to see how they are used to
avoid writing additional parsing code unnecessarily.

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

When a test invokes a particular header command, it is expected that some
behavior will change as a result. What behavior, obviously, will depend on the
purpose of the header command. In the case of `failure-status`, the behavior
that changes is that `compiletest` expects the failure code defined by the
header command invoked in the test, rather than the default value.

Although specific to `failure-status` (as every header command will have a
different implementation in order to invoke behavior change) perhaps it is
helpful to see the behavior change implementation of one case, simply as an
example. To implement `failure-status`, the `check_correct_failure_status()`
function found in the `TestCx` implementation block, located in
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

Note the use of `self.props.failure_status` to access the header command
property. In tests which do not specify the failure status header command,
`self.props.failure_status` will evaluate to the default value of 101 at the
time of this writing. But for a test which specifies a header command of, for
example, `// failure-status: 1`, `self.props.failure_status` will evaluate to
1, as `parse_failure_status()` will have overridden the `TestProps` default
value, for that test specifically.

[`src/tools/compiletest/src/header.rs`]: https://github.com/rust-lang/rust/tree/master/src/tools/compiletest/src/header.rs
[`src/tools/compiletest/src/common.rs`]: https://github.com/rust-lang/rust/tree/master/src/tools/compiletest/src/common.rs
[`src/tools/compiletest/src/runtest.rs`]: https://github.com/rust-lang/rust/tree/master/src/tools/compiletest/src/runtest.rs
