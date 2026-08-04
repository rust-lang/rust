# `llvm-target-feature`

---

Forwards target feature strings directly to LLVM, bypassing Rust's known-feature table.

Registered as a target modifier.

Does not affect `cfg(target_feature)`.

Ignored by non-LLVM backends.

```sh
rustc -Zllvm-target-feature=+prefer-256-bit main.rs
rustc -Zllvm-target-feature=+prefer-256-bit,+fast-gather main.rs
rustc -Zllvm-target-feature=+prefer-256-bit,-avx512f main.rs
```
