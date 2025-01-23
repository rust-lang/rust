# Stabilization report template

> **What is this?** This is a template to use for [stabilization reports](./stabilization_guide.md). It consists of a series of questions that aim to provide the information most commonly needed and to help reviewers be more likely to identify potential problems up front. Not all parts of the template will apply to all stabilizations. Feel free to put N/A if a question doesn't seem to apply to your case.

## General design

### What is the RFC for this feature and what changes have occurred to the user-facing design since the RFC was finalized?

### What behavior are we committing to that has been controversial? Summarize the major arguments pro/con.

### Are there extensions to this feature that remain unstable? How do we know that we are not accidentally committing to those?

## Has a call-for-testing period been conducted? If so, what feedback was received?

## Implementation quality

### Summarize the major parts of the implementation and provide links into the code (or to PRs)

An example for async closures: https://rustc-dev-guide.rust-lang.org/coroutine-closures.html

### Summarize existing test coverage of this feature

- What does the test coverage landscape for this feature look like?
  - (Positive/negative) Behavioral tests?
  - (Positive/negative) Interface tests? (e.g. compiler cli interface)
  - Maybe link to test folders or individual tests (ui/codegen/assembly/run-make tests, etc.)
  - Are there any (intentional/unintentional) gaps in test coverage?

### What outstanding bugs in the issue tracker involve this feature? Are they stabilization-blocking?

### What FIXMEs are still in the code for that feature and why is it ok to leave them there?

### Summarize contributors to the feature by name for recognition and assuredness that people involved in the feature agree with stabilization 

### Which tools need to be adjusted to support this feature. Has this work been done?

*Consider rustdoc, clippy, rust-analyzer, rustfmt, rustup, docs.rs.*

## Type system and execution rules

### What compilation-time checks are done that are needed to prevent undefined behavior?

(Be sure to link to tests demonstrating that these tests are being done.)

### Can users use this feature to introduce undefined behavior, or use this feature to break the abstraction of Rust and expose the underlying assembly-level implementation? (Describe.)

### What updates are needed to the reference/specification? (link to PRs when they exist)

## Common interactions

### Does this feature introduce new expressions and can they produce temporaries? What are the lifetimes of those temporaries?

### What other unstable features may be exposed by this feature?

