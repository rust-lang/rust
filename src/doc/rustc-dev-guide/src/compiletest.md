# `compiletest`

## Introduction

`compiletest` is the main test harness of the Rust test suite.  It allows
test authors to organize large numbers of tests (the Rust compiler has many
thousands), efficient test execution (parallel execution is supported), and
allows the test author to configure behavior and expected results of both
individual and groups of tests.

`compiletest` tests may check test code for success, for failure or in some
cases, even failure to compile.  Tests are typically organized as a Rust source
file with annotations in comments before and/or within the test code, which
serve to direct `compiletest` on if or how to run the test, what behavior to
expect, and more.  If you are unfamiliar with the compiler testing framework,
see [this chapter](./tests/intro.html) for additional background.

The tests themselves are typically (but not always) organized into
"suites" – for example, `run-pass`, a folder representing tests that should
succeed, `run-fail`, a folder holding tests that should compile successfully,
but return a failure (non-zero status), `compile-fail`, a folder holding tests
that should fail to compile, and many more.  The various suites are defined in
[src/tools/compiletest/src/common.rs][common] in the `pub struct Config`
declaration.  And a very good introduction to the different suites of compiler
tests along with details about them can be found in [Adding new
tests](./tests/adding.html).

## Adding a new test file

Briefly, simply create your new test in the appropriate location under
[src/test][test]. No registration of test files is necessary as `compiletest`
will scan the [src/test][test] subfolder recursively, and will execute any Rust
source files it finds as tests.  See [`Adding new tests`](./tests/adding.html)
for a complete guide on how to adding new tests.

## Header Commands

Source file annotations which appear in comments near the top of the source
file *before* any test code are known as header commands.  These commands can
instruct `compiletest` to ignore this test, set expectations on whether it is
expected to succeed at compiling, or what the test's return code is expected to
be.  Header commands (and their inline counterparts, Error Info commands) are
described more fully
[here](./tests/adding.html#header-commands-configuring-rustc).

### Adding a new header command

Header commands are defined in the `TestProps` struct in
[src/tools/compiletest/src/header.rs][header].  At a high level, there are
dozens of test properties defined here, all set to default values in the
`TestProp` struct's `impl` block. Any test can override this default value by
specifying the property in question as header command as a comment (`//`) in
the test source file, before any source code.

#### Using a header command

Here is an example, specifying the `must-compile-successfully` header command,
which takes no arguments, followed by the `failure-status` header command,
which takes a single argument (which, in this case is a value of 1).
`failure-status` is instructing `compiletest` to expect a failure status of 1
(rather than the current Rust default of 101 at the time of this writing).  The
header command and the argument list (if present) are typically separated by a
colon:

```rust,ignore
// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// must-compile-successfully
// failure-status: 1

#![feature(termination_trait)]

use std::io::{Error, ErrorKind};

fn main() -> Result<(), Box<Error>> {
    Err(Box::new(Error::new(ErrorKind::Other, "returned Box<Error> from main()")))
}
```

#### Adding a new header command property

One would add a new header command if there is a need to define some test
property or behavior on an individual, test-by-test basis.  A header command
property serves as the header command's backing store (holds the command's
current value) at runtime.

To add a new header command property:
    1. Look for the `pub struct TestProps` declaration in
       [src/tools/compiletest/src/header.rs][header] and add the new public
       property to the end of the declaration.
    2. Look for the `impl TestProps` implementation block immediately following
       the struct declaration and initialize the new property to its default
       value.

#### Adding a new header command parser

When `compiletest` encounters a test file, it parses the file a line at a time
by calling every parser defined in the `Config` struct's implementation block,
also in [src/tools/compiletest/src/header.rs][header] (note the `Config`
struct's declaration block is found in
[src/tools/compiletest/src/common.rs][common].  `TestProps`'s `load_from()`
method will try passing the current line of text to each parser, which, in turn
typically checks to see if the line begins with a particular commented (`//`)
header command such as `// must-compile-successfully` or `// failure-status`.
Whitespace after the comment marker is optional.

Parsers will override a given header command property's default value merely by
being specified in the test file as a header command or by having a parameter
value specified in the test file, depending on the header command.

Parsers defined in `impl Config` are typically named `parse_<header_command>`
(note kebab-case `<header-command>` transformed to snake-case
`<header_command>`).  `impl Config` also defines several 'low-level' parsers
which make it simple to parse common patterns like simple presence or not
(`parse_name_directive()`), header-command:parameter(s)
(`parse_name_value_directive()`), optional parsing only if a particular `cfg`
attribute is defined (`has_cfg_prefix()`) and many more.  The low-level parsers
are found near the end of the `impl Config` block; be sure to look through them
and their associated parsers immediately above to see how they are used to
avoid writing additional parsing code unneccessarily.

As a concrete example, here is the implementation for the
`parse_failure_status()` parser, in
[src/tools/compiletest/src/header.rs][header]:

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

## Implementing the behavior change

When a test invokes a particular header command, it is expected that some
behavior will change as a result.  What behavior, obviously, will depend on the
purpose of the header command.  In the case of `failure-status`, the behavior
that changes is that `compiletest` expects the failure code defined by the
header command invoked in the test, rather than the default value.

Although specific to `failure-status` (as every header command will have a
different implementation in order to invoke behavior change) perhaps it is
helpful to see the behavior change implementation of one case, simply as an
example.  To implement `failure-status`, the `check_correct_failure_status()`
function found in the `TestCx` implementation block, located in
[src/tools/compiletest/src/runtest.rs](https://github.com/rust-lang/rust/tree/master/src/tools/compiletest/src/runtest.rs),
was modified as per below:

```diff
@@ -295,11 +295,14 @@ impl<'test> TestCx<'test> {
     }

     fn check_correct_failure_status(&self, proc_res: &ProcRes) {
-        // The value the rust runtime returns on failure
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
property.  In tests which do not specify the failure status header command,
`self.props.failure_status` will evaluate to the default value of 101 at the
time of this writing.  But for a test which specifies a header command of, for
example, `// failure-status: 1`, `self.props.failure_status` will evaluate to
1, as `parse_failure_status()` will have overridden the `TestProps` default
value, for that test specifically.

[test]: https://github.com/rust-lang/rust/tree/master/src/test
[header]: https://github.com/rust-lang/rust/tree/master/src/tools/compiletest/src/header.rs
[common]: https://github.com/rust-lang/rust/tree/master/src/tools/compiletest/src/common.rs
