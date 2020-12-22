---
name: Library Tracking Issue
about: A tracking issue for an unstable library feature.
title: Tracking Issue for XXX
labels: C-tracking-issue, T-libs
---
<!--
Thank you for creating a tracking issue!

Tracking issues are for tracking a feature from implementation to stabilization.

Make sure to include the relevant RFC for the feature if it has one.

If the new feature is small, it may be fine to skip the RFC process. In that
case, you can use use `issue = "none"` in your initial implementation PR. The
reviewer will ask you to open a tracking issue if they agree your feature can be
added without an RFC.
-->

Feature gate: `#![feature(...)]`

This is a tracking issue for ...

<!--
Include a short description of the feature.
-->

### Public API

<!--
For most library features, it'd be useful to include a summarized version of the public API.
(E.g. just the public function signatures without their doc comments or implementation.)
-->

```rust
...
```

### Steps / History

<!--
In the simplest case, this is a PR implementing the feature followed by a PR
that stabilises the feature. However it's not uncommon for the feature to be
changed before stabilization. For larger features, the implementation could be
split up in multiple steps.
-->

- [ ] Implementation: ...
- [ ] Stabilization PR

### Unresolved Questions

<!--
Include any open questions that need to be answered before the feature can be
stabilised. If multiple (unrelated) big questions come up, it can be a good idea
to open a separate issue for each, to make it easier to keep track of the
discussions.

It's useful to link any relevant discussions and conclusions (whether on GitHub,
Zulip, or the internals forum) here.
-->

- None yet.
