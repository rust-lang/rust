---
name: Bug Report (False Positive)
about: Create a bug report about a wrongly emitted lint warning
labels: L-bug, L-false-positive
---
<!--
Thank you for filing a bug report! 🐛 Please provide a short summary of the bug,
along with any information you feel relevant to replicating the bug.
-->
Lint name:


I tried this code:

```rust
<code>
```

I expected to see this happen: *explanation*

Instead, this happened: *explanation*

### Meta

- `cargo clippy -V`: e.g. clippy 0.0.212 (f455e46 2020-06-20)
- `rustc -Vv`:
  ```
  rustc 1.46.0-nightly (f455e46ea 2020-06-20)
  binary: rustc
  commit-hash: f455e46eae1a227d735091091144601b467e1565
  commit-date: 2020-06-20
  host: x86_64-unknown-linux-gnu
  release: 1.46.0-nightly
  LLVM version: 10.0
  ```
