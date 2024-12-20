---
name: Internal Compiler Error
about: Create a report for an internal compiler error in rustc.
labels: C-bug, I-ICE, T-compiler
---
<!--
Thank you for finding an Internal Compiler Error! 🧊  If possible, try to provide
a minimal verifiable example. You can read "Rust Bug Minimization Patterns" for
how to create smaller examples.
http://blog.pnkfx.org/blog/2019/11/18/rust-bug-minimization-patterns/

Please check if your ICE has already been reported by searching for parts of the ICE-message
e.g. "invalid asymmetric binary op Lt" or "assertion failed: self.let_source != LetSource::None"
and the source location of the ICE, such as "compiler/rustc_const_eval/src/interpret/operator.rs"
in the bugtracker and leave a comment with your reproducer if you find tickets corresponding to your ICE.
If you are unsure and decide to open a new issue, please leave a link to issues you suspect might be related or duplicates.
-->

### Code

```Rust
<code>
```


### Meta
<!--
If you're using the stable version of the compiler, you should also check if the
bug also exists in the beta or nightly versions.
-->

`rustc --version --verbose`:
```
<version>
```

### Error output

```
<output>
```

<!--
Include a backtrace in the code block by setting `RUST_BACKTRACE=1` in your
environment. E.g. `RUST_BACKTRACE=1 cargo build`.
-->
<details><summary><strong>Backtrace</strong></summary>
<p>

```
<backtrace>
```

</p>
</details>
