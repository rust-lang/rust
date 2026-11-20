# `print=wasm-proc-macro-tuple`

The tracking issue for this feature is: [#160389](https://github.com/rust-lang/rust/issues/160389).

------------------------

This option of the `--print` flag produces the target for which wasm proc-macros should be compiled.

Intended to be used like this:

```bash
rustc --print=wasm-proc-macro-tuple -Zunstable-options
```
