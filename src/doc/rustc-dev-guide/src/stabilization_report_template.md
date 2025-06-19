# Stabilization report template

> **What is this?**
>
> This is a template to use for [stabilization reports](./stabilization_guide.md) of **language features**. It consists of a series of questions that aim to provide the information most commonly needed and to help reviewers be more likely to identify potential problems up front. Not all parts of the template will apply to all stabilizations. Feel free to put N/A if a question doesn't seem to apply to your case.
>
> You can copy the following template after the separator and edit it as Markdown, replacing the *TODO* placeholders with answers. 

---

> ## General design

>  ### What is the RFC for this feature and what changes have occurred to the user-facing design since the RFC was finalized?

*TODO*

>  ### What behavior are we committing to that has been controversial? Summarize the major arguments pro/con.

*TODO*

> ### Are there extensions to this feature that remain unstable? How do we know that we are not accidentally committing to those?

*TODO*

> ## Has a Call for Testing period been conducted? If so, what feedback was received?
>
> Does any OSS nightly users use this feature? For instance, a useful indication might be "search <grep.app> for `#![feature(FEATURE_NAME)]` and had `N` results".

*TODO*

> ## Implementation quality

*TODO*

> ### Summarize the major parts of the implementation and provide links into the code (or to PRs)
>
> An example for async closures: <https://rustc-dev-guide.rust-lang.org/coroutine-closures.html>.

*TODO*

> ### Summarize existing test coverage of this feature
>
> Consider what the "edges" of this feature are.  We're particularly interested in seeing tests that assure us about exactly what nearby things we're not stabilizing.
>
> Within each test, include a comment at the top describing the purpose of the test and what set of invariants it intends to demonstrate. This is a great help to those reviewing the tests at stabilization time.
>
> - What does the test coverage landscape for this feature look like?
>   - Tests for compiler errors when you use the feature wrongly or make mistakes?
>   - Tests for the feature itself:
>       - Limits of the feature (so failing compilation)
>       - Exercises of edge cases of the feature
>       - Tests that checks the feature works as expected (where applicable, `//@ run-pass`).
>   - Are there any intentional gaps in test coverage?
>
> Link to test folders or individual tests (ui/codegen/assembly/run-make tests, etc.).

*TODO*

> ### What outstanding bugs in the issue tracker involve this feature? Are they stabilization-blocking?

*TODO*

> ### What FIXMEs are still in the code for that feature and why is it ok to leave them there?

*TODO*

> ### Summarize contributors to the feature by name for recognition and assuredness that people involved in the feature agree with stabilization 

*TODO*

> ### Which tools need to be adjusted to support this feature. Has this work been done?
>  
> Consider rustdoc, clippy, rust-analyzer, rustfmt, rustup, docs.rs.

*TODO*

> ## Type system and execution rules

> ### What compilation-time checks are done that are needed to prevent undefined behavior?
>  
>  (Be sure to link to tests demonstrating that these tests are being done.)

*TODO*

> ### Does the feature's implementation need checks to prevent UB or is it sound by default and needs opt in in places to perform the dangerous/unsafe operations? If it is not sound by default, what is the rationale?

*TODO*

> ### Can users use this feature to introduce undefined behavior, or use this feature to break the abstraction of Rust and expose the underlying assembly-level implementation? (Describe.)

*TODO*

> ### What updates are needed to the reference/specification? (link to PRs when they exist)

*TODO*

> ## Common interactions

> ### Does this feature introduce new expressions and can they produce temporaries? What are the lifetimes of those temporaries?

*TODO*

> ### What other unstable features may be exposed by this feature?

*TODO*
