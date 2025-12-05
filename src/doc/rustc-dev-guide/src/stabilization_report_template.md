# Stabilization report template

## What is this?

This is a template for [stabilization reports](./stabilization_guide.md) of **language features**. The questions aim to solicit the details most often needed. These details help reviewers to identify potential problems upfront. Not all parts of the template will apply to every stabilization. If a question doesn't apply, explain briefly why.

Copy everything after the separator and edit it as Markdown. Replace each *TODO* with your answer.

---

# Stabilization report

## Summary

> Remind us what this feature is and what value it provides. Tell the story of what led up to this stabilization.
>
> E.g., see:
>
> - [Stabilize AFIT/RPITIT](https://web.archive.org/web/20250329190642/https://github.com/rust-lang/rust/pull/115822)
> - [Stabilize RTN](https://web.archive.org/web/20250321214601/https://github.com/rust-lang/rust/pull/138424)
> - [Stabilize ATPIT](https://web.archive.org/web/20250124214256/https://github.com/rust-lang/rust/pull/120700)
> - [Stabilize opaque type precise capturing](https://web.archive.org/web/20250312173538/https://github.com/rust-lang/rust/pull/127672)

*TODO*

Tracking:

- *TODO* (Link to tracking issue.)

Reference PRs:

- *TODO* (Link to Reference PRs.)

cc @rust-lang/lang @rust-lang/lang-advisors

### What is stabilized

> Describe each behavior being stabilized and give a short example of code that will now be accepted.

```rust
todo!()
```

### What isn't stabilized

> Describe any parts of the feature not being stabilized. Talk about what we might want to do later and what doors are being left open for that. If what we're not stabilizing might lead to surprises for users, talk about that in particular.

## Design

### Reference

> What updates are needed to the Reference? Link to each PR. If the Reference is missing content needed for describing this feature, discuss that.

- *TODO*

### RFC history

> What RFCs have been accepted for this feature?

- *TODO*

### Answers to unresolved questions

> What questions were left unresolved by the RFC? How have they been answered? Link to any relevant lang decisions.

*TODO*

### Post-RFC changes

> What other user-visible changes have occurred since the RFC was accepted? Describe both changes that the lang team accepted (and link to those decisions) as well as changes that are being presented to the team for the first time in this stabilization report.

*TODO*

### Key points

> What decisions have been most difficult and what behaviors to be stabilized have proved most contentious? Summarize the major arguments on all sides and link to earlier documents and discussions.

*TODO*

### Nightly extensions

> Are there extensions to this feature that remain unstable? How do we know that we are not accidentally committing to those?

*TODO*

### Doors closed

> What doors does this stabilization close for later changes to the language? E.g., does this stabilization make any other RFCs, lang experiments, or known in-flight proposals more difficult or impossible to do later?

## Feedback

### Call for testing

> Has a "call for testing" been done? If so, what feedback was received?

*TODO*

### Nightly use

> Do any known nightly users use this feature? Counting instances of `#![feature(FEATURE_NAME)]` on GitHub with grep might be informative.

*TODO*

## Implementation

### Major parts

> Summarize the major parts of the implementation and provide links into the code and to relevant PRs.
>
> See, e.g., this breakdown of the major parts of async closures:
>
> - <https://rustc-dev-guide.rust-lang.org/coroutine-closures.html>

*TODO*

### Coverage

> Summarize the test coverage of this feature.
>
> Consider what the "edges" of this feature are. We're particularly interested in seeing tests that assure us about exactly what nearby things we're not stabilizing. Tests should of course comprehensively demonstrate that the feature works. Think too about demonstrating the diagnostics seen when common mistakes are made and the feature is used incorrectly.
>
> Within each test, include a comment at the top describing the purpose of the test and what set of invariants it intends to demonstrate. This is a great help to our review.
>
> Describe any known or intentional gaps in test coverage.
>
> Contextualize and link to test folders and individual tests.

*TODO*

### Outstanding bugs

> What outstanding bugs involve this feature? List them. Should any block the stabilization? Discuss why or why not.

*TODO*

- *TODO*
- *TODO*
- *TODO*

### Outstanding FIXMEs

> What FIXMEs are still in the code for that feature and why is it OK to leave them there?

*TODO*

### Tool changes

> What changes must be made to our other tools to support this feature. Has this work been done? Link to any relevant PRs and issues.

- [ ] rustfmt
  - *TODO*
- [ ] rust-analyzer
  - *TODO*
- [ ] rustdoc (both JSON and HTML)
  - *TODO*
- [ ] cargo
  - *TODO*
- [ ] clippy
  - *TODO*
- [ ] rustup
  - *TODO*
- [ ] docs.rs
  - *TODO*

*TODO*

### Breaking changes

> If this stabilization represents a known breaking change, link to the crater report, the analysis of the crater report, and to all PRs we've made to ecosystem projects affected by this breakage. Discuss any limitations of what we're able to know about or to fix.

*TODO*

Crater report:

- *TODO*

Crater analysis:

- *TODO*

PRs to affected crates:

- *TODO*
- *TODO*
- *TODO*

## Type system, opsem

### Compile-time checks

> What compilation-time checks are done that are needed to prevent undefined behavior?
>
> Link to tests demonstrating that these checks are being done.

*TODO*

- *TODO*
- *TODO*
- *TODO*

### Type system rules

> What type system rules are enforced for this feature and what is the purpose of each?

*TODO*

### Sound by default?

> Does the feature's implementation need specific checks to prevent UB, or is it sound by default and need specific opt-in to perform the dangerous/unsafe operations? If it is not sound by default, what is the rationale?

*TODO*

### Breaks the AM?

> Can users use this feature to introduce undefined behavior, or use this feature to break the abstraction of Rust and expose the underlying assembly-level implementation? Describe this if so.

*TODO*

## Common interactions

### Temporaries

> Does this feature introduce new expressions that can produce temporaries? What are the scopes of those temporaries?

*TODO*

### Drop order

> Does this feature raise questions about the order in which we should drop values? Talk about the decisions made here and how they're consistent with our earlier decisions.

*TODO*

### Pre-expansion / post-expansion

> Does this feature raise questions about what should be accepted pre-expansion (e.g. in code covered by `#[cfg(false)]`) versus what should be accepted post-expansion? What decisions were made about this?

*TODO*

### Edition hygiene

> If this feature is gated on an edition, how do we decide, in the context of the edition hygiene of tokens, whether to accept or reject code. E.g., what token do we use to decide?

*TODO*

### SemVer implications

> Does this feature create any new ways in which library authors must take care to prevent breaking downstreams when making minor-version releases? Describe these. Are these new hazards "major" or "minor" according to [RFC 1105](https://rust-lang.github.io/rfcs/1105-api-evolution.html)?

*TODO*

### Exposing other features

> Are there any other unstable features whose behavior may be exposed by this feature in any way? What features present the highest risk of that?

*TODO*

## History

> List issues and PRs that are important for understanding how we got here.

- *TODO*
- *TODO*
- *TODO*

## Acknowledgments

> Summarize contributors to the feature by name for recognition and so that those people are notified about the stabilization. Does anyone who worked on this *not* think it should be stabilized right now? We'd like to hear about that if so.

*TODO*

## Open items

> List any known items that have not yet been completed and that should be before this is stabilized.

- [ ] *TODO*
- [ ] *TODO*
- [ ] *TODO*
