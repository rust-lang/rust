---
name: Tracking Issue
about: A tracking issue for an accepted feature or RFC in Rust.
title: Tracking Issue for XXX
labels: C-tracking-issue
---
<!--
NOTE: For library features, please use the "Library Tracking Issue" template instead.

Thank you for creating a tracking issue! 📜 Tracking issues are for tracking a
feature from implementation to stabilisation. Make sure to include the relevant
RFC for the feature if it has one. Otherwise provide a short summary of the
feature and link any relevant PRs or issues, and remove any sections that are
not relevant to the feature.

Remember to add team labels to the tracking issue.
For a language team feature, this would e.g., be `T-lang`.
Such a feature should also be labeled with e.g., `F-my_feature`.
This label is used to associate issues (e.g., bugs and design questions) to the feature.
-->

This is a tracking issue for the RFC "XXX" (rust-lang/rfcs#NNN).
The feature gate for the issue is `#![feature(FFF)]`.

### About tracking issues

Tracking issues are used to record the overall progress of implementation.
They are also used as hubs connecting to other relevant issues, e.g., bugs or open design questions.
A tracking issue is however *not* meant for large scale discussion, questions, or bug reports about a feature.
Instead, open a dedicated issue for the specific matter and add the relevant feature gate label.
Discussion comments will get marked as off-topic or deleted.
Repeated discussions on the tracking issue may lead to the tracking issue getting locked.

### Steps
<!--
Include each step required to complete the feature. Typically this is a PR
implementing a feature, followed by a PR that stabilises the feature. However
for larger features an implementation could be broken up into multiple PRs.
-->

- [ ] Implement the RFC (cc @rust-lang/XXX -- can anyone write up mentoring
      instructions?)
- [ ] Adjust documentation ([see instructions on rustc-dev-guide][doc-guide])
- [ ] Style updates for any new syntax ([nightly-style-procedure])
  - [ ] Style team decision on new formatting
  - [ ] Formatting for new syntax has been added to the [Style Guide]
  - [ ] (non-blocking) Formatting has been implemented in `rustfmt`
- [ ] Stabilization PR ([see instructions on rustc-dev-guide][stabilization-guide])

[stabilization-guide]: https://rustc-dev-guide.rust-lang.org/stabilization_guide.html#stabilization-pr
[doc-guide]: https://rustc-dev-guide.rust-lang.org/stabilization_guide.html#documentation-prs
[nightly-style-procedure]: https://github.com/rust-lang/style-team/blob/main/nightly-style-procedure.md 
[Style Guide]: https://github.com/rust-lang/rust/tree/master/src/doc/style-guide

### Unresolved Questions
<!--
Include any open questions that need to be answered before the feature can be
stabilised.
-->

XXX --- list all the "unresolved questions" found in the RFC to ensure they are
not forgotten

### Implementation history

<!--
Include a list of all the PRs that were involved in implementing the feature.
-->
