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
// core::magic

pub struct Magic;

impl Magic {
    pub fn magic(self);
}
```

### Steps / History

<!--
For larger features, more steps might be involved.
If the feature is changed later, please add those PRs here as well.
-->

- [ ] Implementation: #...
- [ ] Final commenting period (FCP)
- [ ] Stabilization PR

<!--
Once the feature has gone through a few release cycles and there are no
unresolved questions left, the feature might be ready for stabilization.

If this feature didn't go through the RFC process, a final commenting period
(FCP) is always needed before stabilization. This works as follows:

A library team member can kick off the stabilization process, at which point
the rfcbot will ask all the team members to verify they agree with
stabilization. Once enough members agree and there are no concerns, the final
commenting period begins: this issue will be marked as such and will be listed
in the next This Week in Rust newsletter. If no blocking concerns are raised in
that period of 10 days, a stabilzation PR can be opened by anyone.
-->

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
