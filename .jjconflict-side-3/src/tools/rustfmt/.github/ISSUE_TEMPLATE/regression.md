---
name: Regression
about: Report something that unexpectedly changed between rustfmt versions.
labels: C-bug, regression-untriaged
---
<!--
Thank you for filing a regression report! 🐛 A regression is something that changed between
versions of rustfmt but was not supposed to.

Please provide a short summary of the regression, along with any information you
feel is relevant to replicate it.
-->

## Summary

<!--
Please include a reproducer for the bug you are describing.
If possible, try to provide a Minimal, Complete and Verifiable example.
You can read "Rust Bug Minimization Patterns" for how to create smaller examples.
http://blog.pnkfx.org/blog/2019/11/18/rust-bug-minimization-patterns/
-->

I tried to format this code:

```rust
<code>
```

### Expected behavior

I expected to see this happen: *explanation*

### Actual behavior

Instead, this happened: *explanation*


## Meta

### Version it worked on

<!--
Provide the most recent version this worked on, for example:

It most recently worked on: Rust 1.47
-->

It most recently worked on: <!-- version -->

### Version with regression

<!--
Provide the version you are using that has the regression.
-->

`rustc --version --verbose`:
```
<version>
```

<!--
If you know when this regression occurred, please add a line like below, replacing `{channel}`
with one of stable, beta, or nightly.

@rustbot modify labels: +regression-from-stable-to-{channel} -regression-untriaged
-->
