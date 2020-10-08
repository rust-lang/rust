---
name: Regression
about: Report something that unexpectedly changed between Rust versions.
labels: C-bug regression-untriaged
---
<!--
Thank you for filing a regression report! 🐛 A regression is something that changed between versions of Rust but was not supposed to.

Please provide a short summary of the regression, along with any information you feel is relevant to replicate it.
-->

### Code

I tried this code:

```rust
<code>
```

I expected to see this happen: *explanation*

Instead, this happened: *explanation*

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
Did the compiler crash? If so, please provide a backtrace.
-->

### Backtrace
<!--
Include a backtrace in the code block by setting `RUST_BACKTRACE=1` in your
environment. E.g. `RUST_BACKTRACE=1 cargo build`.
-->
<details><summary>Backtrace</summary>
<p>

```
<backtrace>
```

</p>
</details>

<!--
If you know when this regression occurred, please add a line like below, replacing `{channel}` with one of stable, beta, or nightly.

@rustbot modify labels: +regression-from-stable-to-{channel} -regression-untriaged
-->
