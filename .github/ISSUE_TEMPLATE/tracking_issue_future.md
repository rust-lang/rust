---
name: Future Incompatibility Tracking Issue
about: A tracking issue for a future-incompatible lint
title: Tracking Issue for future-incompatibility lint XXX
labels: C-tracking-issue C-future-incompatibility T-compiler A-lints
---
<!--
Thank you for creating a future-incompatible tracking issue! 📜 These issues
are for lints that implement a future-incompatible warning.

Remember to add team labels to the tracking issue.
For something that affects the language, this would be `T-lang`, and for libs
it would be `T-libs-api`.
Also check for any `A-` labels to add.
-->

This is the **tracking issue** for the `YOUR_LINT_NAME_HERE` future-compatibility warning and other related errors. The goal of this page is describe why this change was made and how you can fix code that is affected by it. It also provides a place to ask questions or register a complaint if you feel the change should not be made. For more information on the policy around future-compatibility warnings, see our [breaking change policy guidelines][guidelines].

[guidelines]: https://rustc-dev-guide.rust-lang.org/bug-fix-procedure.html

### What is the warning for?

*Describe the conditions that trigger the warning.*

### Why was this change made?

*Explain why this change was made. If there is additional context, like an MCP, link it here.*

### Example

```rust
// Include an example here.
```

### Recommendations

*Give some recommendations on how a user can avoid the lint.*

### When will this warning become a hard error?

*If known, describe the future plans. For example, how long you anticipate this being a warning, or if there are other factors that will influence the anticipated closure.*

### Steps

- [ ] Implement the lint
- [ ] Raise lint level to deny
- [ ] Make lint report in dependencies
- [ ] Switch to a hard error

### Implementation history

<!--
Include a list of all the PRs that were involved in implementing the lint.
-->
