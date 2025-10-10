# Adding new tests

**In general, we expect every PR that fixes a bug in rustc to come accompanied
by a regression test of some kind.** This test should fail in master but pass
after the PR. These tests are really useful for preventing us from repeating the
mistakes of the past.

The first thing to decide is which kind of test to add. This will depend on the
nature of the change and what you want to exercise. Here are some rough
guidelines:

- The majority of compiler tests are done with [compiletest].
  - The majority of compiletest tests are [UI](ui.md) tests in the [`tests/ui`]
    directory.
- Changes to the standard library are usually tested within the standard library
  itself.
  - The majority of standard library tests are written as doctests, which
    illustrate and exercise typical API behavior.
  - Additional [unit tests](intro.md#package-tests) should go in
    `library/${crate}/tests` (where `${crate}` is usually `core`, `alloc`, or
    `std`).
- If the code is part of an isolated system, and you are not testing compiler
  output, consider using a [unit or integration test](intro.md#package-tests).
- Need to run rustdoc? Prefer a `rustdoc` or `rustdoc-ui` test. Occasionally
  you'll need `rustdoc-js` as well.
- Other compiletest test suites are generally used for special purposes:
  - Need to run gdb or lldb? Use the `debuginfo` test suite.
  - Need to inspect LLVM IR or MIR IR? Use the `codegen` or `mir-opt` test
    suites.
  - Need to inspect the resulting binary in some way? Or if all the other test
    suites are too limited for your purposes? Then use `run-make`.
    - Use `run-make-cargo` if you need to exercise in-tree `cargo` in conjunction
      with in-tree `rustc`.
  - Check out the [compiletest] chapter for more specialized test suites.

After deciding on which kind of test to add, see [best
practices](best-practices.md) for guidance on how to author tests that are easy
to work with that stand the test of time (i.e. if a test fails or need to be
modified several years later, how can we make it easier for them?).

[compiletest]: compiletest.md
[`tests/ui`]: https://github.com/rust-lang/rust/tree/master/tests/ui/

## UI test walkthrough

The following is a basic guide for creating a [UI test](ui.md), which is one of
the most common compiler tests. For this tutorial, we'll be adding a test for an
async error message.

### Step 1: Add a test file

The first step is to create a Rust source file somewhere in the [`tests/ui`]
tree. When creating a test, do your best to find a good location and name (see
[Test organization](ui.md#test-organization) for more). Since naming is the
hardest part of development, everything should be downhill from here!

Let's place our async test at `tests/ui/async-await/await-without-async.rs`:

```rust,ignore
// Provide diagnostics when the user writes `await` in a non-`async` function.
//@ edition:2018

async fn foo() {}

fn bar() {
    foo().await
}

fn main() {}
```

A few things to notice about our test:

- The top should start with a short comment that [explains what the test is
  for](#explanatory_comment).
- The `//@ edition:2018` comment is called a [directive](directives.md) which
  provides instructions to compiletest on how to build the test. Here we need to
  set the edition for `async` to work (the default is edition 2015).
- Following that is the source of the test. Try to keep it succinct and to the
  point. This may require some effort if you are trying to minimize an example
  from a bug report.
- We end this test with an empty `fn main` function. This is because the default
  for UI tests is a `bin` crate-type, and we don't want the "main not found"
  error in our test. Alternatively, you could add `#![crate_type="lib"]`.

### Step 2: Generate the expected output

The next step is to create the expected output snapshots from the compiler. This
can be done with the `--bless` option:

```sh
./x test tests/ui/async-await/await-without-async.rs --bless
```

This will build the compiler (if it hasn't already been built), compile the
test, and place the output of the compiler in a file called
`tests/ui/async-await/await-without-async.stderr`.

However, this step will fail! You should see an error message, something like
this:

> error: /rust/tests/ui/async-await/await-without-async.rs:7: unexpected
> error: '7:10: 7:16: `await` is only allowed inside `async` functions and
> blocks E0728'

This is because the stderr contains errors which were not matched by error
annotations in the source file.

### Step 3: Add error annotations

Every error needs to be annotated with a comment in the source with the text of
the error. In this case, we can add the following comment to our test file:

```rust,ignore
fn bar() {
    foo().await
    //~^ ERROR `await` is only allowed inside `async` functions and blocks
}
```

The `//~^` squiggle caret comment tells compiletest that the error belongs to
the *previous* line (more on this in the [Error
annotations](ui.md#error-annotations) section).

Save that, and run the test again:

```sh
./x test tests/ui/async-await/await-without-async.rs
```

It should now pass, yay!

### Step 4: Review the output

Somewhat hand-in-hand with the previous step, you should inspect the `.stderr`
file that was created to see if it looks like how you expect. If you are adding
a new diagnostic message, now would be a good time to also consider how readable
the message looks overall, particularly for people new to Rust.

Our example `tests/ui/async-await/await-without-async.stderr` file should look
like this:

```text
error[E0728]: `await` is only allowed inside `async` functions and blocks
  --> $DIR/await-without-async.rs:7:10
   |
LL | fn bar() {
   |    --- this is not `async`
LL |     foo().await
   |          ^^^^^^ only allowed inside `async` functions and blocks

error: aborting due to previous error

For more information about this error, try `rustc --explain E0728`.
```

You may notice some things look a little different than the regular compiler
output.

- The `$DIR` removes the path information which will differ between systems.
- The `LL` values replace the line numbers. That helps avoid small changes in
  the source from triggering large diffs. See the
  [Normalization](ui.md#normalization) section for more.

Around this stage, you may need to iterate over the last few steps a few times
to tweak your test, re-bless the test, and re-review the output.

### Step 5: Check other tests

Sometimes when adding or changing a diagnostic message, this will affect other
tests in the test suite. The final step before posting a PR is to check if you
have affected anything else. Running the UI suite is usually a good start:

```sh
./x test tests/ui
```

If other tests start failing, you may need to investigate what has changed and
if the new output makes sense.

You may also need to re-bless the output with the `--bless` flag.

<a id="explanatory_comment"></a>

## Comment explaining what the test is about

The first comment of a test file should **summarize the point of the test**, and
highlight what is important about it. If there is an issue number associated
with the test, include the issue number.

This comment doesn't have to be super extensive. Just something like "Regression
test for #18060: match arms were matching in the wrong order." might already be
enough.

These comments are very useful to others later on when your test breaks, since
they often can highlight what the problem is. They are also useful if for some
reason the tests need to be refactored, since they let others know which parts
of the test were important. Often a test must be rewritten because it no longer
tests what it was meant to test, and then it's useful to know what it *was*
meant to test exactly.
