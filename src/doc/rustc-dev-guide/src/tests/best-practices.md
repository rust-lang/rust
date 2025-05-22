# Best practices for writing tests

This chapter describes best practices related to authoring and modifying tests.
We want to make sure the tests we author are easy to understand and modify, even
several years later, without needing to consult the original author and perform
a bunch of git archeology.

It's good practice to review the test that you authored by pretending that you
are a different contributor who is looking at the test that failed several years
later without much context (this also helps yourself even a few days or months
later!). Then ask yourself: how can I make my life and their lives easier?

To help put this into perspective, let's start with an aside on how to write a
test that makes the life of another contributor as hard as possible.

> **Aside: Simple Test Sabotage Field Manual**
>
> To make the life of another contributor as hard as possible, one might:
>
> - Name the test after an issue number alone without any other context, e.g.
>   `issue-123456.rs`.
> - Have no comments at all on what the test is trying to exercise, no links to
>   relevant context.
> - Include a test that is massive (that can otherwise be minimized) and
>   contains non-essential pieces which distracts from the core thing the test
>   is actually trying to test.
> - Include a bunch of unrelated syntax errors and other errors which are not
>   critical to what the test is trying to check.
> - Weirdly format the snippets.
> - Include a bunch of unused and unrelated features.
> - Have e.g. `ignore-windows` [compiletest directives] but don't offer any
>   explanation as to *why* they are needed.

## Test naming

Make it easy for the reader to immediately understand what the test is
exercising, instead of having to type in the issue number and dig through github
search for what the test is trying to exercise. This has an additional benefit
of making the test possible to be filtered via `--test-args` as a collection of
related tests.

- Name the test after what it's trying to exercise or prevent regressions of.
- Keep it concise.
- Avoid using issue numbers alone as test names.
- Avoid starting the test name with `issue-xxxxx` prefix as it degrades
  auto-completion.

> **Avoid using only issue numbers as test names**
>
> Prefer including them as links or `#123456` in test comments instead. Or if it
> makes sense to include the issue number, also include brief keywords like
> `macro-external-span-ice-123956.rs`.
>
> ```text
> tests/ui/typeck/issue-123456.rs                              // bad
> tests/ui/typeck/issue-123456-asm-macro-external-span-ice.rs  // bad (for tab completion)
> tests/ui/typeck/asm-macro-external-span-ice-123456.rs        // good
> tests/ui/typeck/asm-macro-external-span-ice.rs               // good
> ```
>
> `issue-123456.rs` does not tell you immediately anything about what the test
> is actually exercising meaning you need to do additional searching. Including
> the issue number in the test name as a prefix makes tab completion less useful
> (if you `ls` a test directory and get a bunch of `issue-xxxxx` prefixes). We
> can link to the issue in a test comment.
>
> ```rs
> //! Check that `asm!` macro including nested macros that come from external
> //! crates do not lead to a codepoint boundary assertion ICE.
> //!
> //! Regression test for <https://github.com/rust-lang/rust/issues/123456>.
> ```
>
> One exception to this rule is [crashes tests]: there it is canonical that
> tests are named only after issue numbers because its purpose is to track
> snippets from which issues no longer ICE/crash, and they would either be
> removed or converted into proper ui/other tests in the fix PRs.

## Test organization

- For most test suites, try to find a semantically meaningful subdirectory to
  home the test.
    - E.g. for an implementation of RFC 2093 specifically, we can group a
      collection of tests under `tests/ui/rfc-2093-infer-outlives/`. For the
      directory name, include what the RFC is about.
- For the [`run-make`] test suite, each `rmake.rs` must be contained within an
  immediate subdirectory under `tests/run-make/`. Further nesting is not
  presently supported. Avoid including issue number in the directory name too,
  include that info in a comment inside `rmake.rs`.

## Test descriptions

To help other contributors understand what the test is about if their changes
lead to the test failing, we should make sure a test has sufficient docs about
its intent/purpose, links to relevant context (incl. issue numbers or other
discussions) and possibly relevant resources (e.g. can be helpful to link to
Win32 APIs for specific behavior).

**Synopsis of a test with good comments**

```rust,ignore
//! Brief summary of what the test is exercising.
//! Example: Regression test for #123456: make sure coverage attribute don't ICE
//!     when applied to non-items.
//!
//! Optional: Remarks on related tests/issues, external APIs/tools, crash
//!     mechanism, how it's fixed, FIXMEs, limitations, etc.
//! Example: This test is like `tests/attrs/linkage.rs`, but this test is
//!     specifically checking `#[coverage]` which exercises a different code
//!     path. The ICE was triggered during attribute validation when we tried
//!     to construct a `def_path_str` but only emitted the diagnostic when the
//!     platform is windows, causing an ICE on unix.
//!
//! Links to relevant issues and discussions. Examples below:
//! Regression test for <https://github.com/rust-lang/rust/issues/123456>.
//! See also <https://github.com/rust-lang/rust/issues/101345>.
//! See discussion at <https://rust-lang.zulipchat.com/#narrow/stream/131828-t-compiler/topic/123456-example-topic>.
//! See [`clone(2)`].
//!
//! [`clone(2)`]: https://man7.org/linux/man-pages/man2/clone.2.html

//@ ignore-windows
// Reason: (why is this test ignored for windows? why not specifically
// windows-gnu or windows-msvc?)

// Optional: Summary of test cases: What positive cases are checked?
// What negative cases are checked? Any specific quirks?

fn main() {
    #[coverage]
    //~^ ERROR coverage attribute can only be applied to function items.
    let _ = {
        // Comment highlighting something that deserves reader attention.
        fn foo() {}
    };
}
```

For how much context/explanation is needed, it is up to the author and
reviewer's discretion. A good rule of thumb is non-trivial things exercised in
the test deserves some explanation to help other contributors to understand.
This may include remarks on:

- How an ICE can get triggered if it's quite elaborate.
- Related issues and tests (e.g. this test is like another test but is kept
  separate because...).
- Platform-specific behaviors.
- Behavior of external dependencies and APIs: syscalls, linkers, tools,
  environments and the likes.

## Test content

- Try to make sure the test is as minimal as possible.
- Minimize non-critical code and especially minimize unnecessary syntax and type
  errors which can clutter stderr snapshots.
- Where possible, use semantically meaningful names (e.g. `fn
  bare_coverage_attributes() {}`).

## Flaky tests

All tests need to strive to be reproducible and reliable. Flaky tests are the
worst kind of tests, arguably even worse than not having the test in the first
place.

- Flaky tests can fail in completely unrelated PRs which can confuse other
  contributors and waste their time trying to figure out if test failure is
  related.
- Flaky tests provide no useful information from its test results other than
  it's flaky and not reliable: if a test passed but it's flakey, did I just get
  lucky? if a test is flakey but it failed, was it just spurious?
- Flaky tests degrade confidence in the whole test suite. If a test suite can
  randomly spuriously fail due to flaky tests, did the whole test suite pass or
  did I just get lucky/unlucky?
- Flaky tests can randomly fail in full CI, wasting previous full CI resources.

## Compiletest directives

See [compiletest directives] for a listing of directives.

- For `ignore-*`/`needs-*`/`only-*` directives, unless extremely obvious,
  provide a brief remark on why the directive is needed. E.g. `"//@ ignore-wasi
  (wasi codegens the main symbol differently)"`.
- When using `//@ ignore-auxiliary`, specify the corresponding main test files,
  e.g. ``//@ ignore-auxiliary (used by `./foo.rs`)``.

## FileCheck best practices

See [LLVM FileCheck guide][FileCheck] for details.

- Avoid matching on specific register numbers or basic block numbers unless
  they're special or critical for the test. Consider using patterns to match
  them where suitable.

> **TODO**
>
> Pending concrete advice.

[compiletest]: ./compiletest.md
[compiletest directives]: ./directives.md
[`run-make`]: ./compiletest.md#run-make-tests
[FileCheck]: https://llvm.org/docs/CommandGuide/FileCheck.html
[crashes tests]: ./compiletest.md#crashes-tests
