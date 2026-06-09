# `print=supported-crate-types`

The tracking issue for this feature is: [#138640](https://github.com/rust-lang/rust/issues/138640).

------------------------

This option of the `--print` flag produces a list of crate types (delimited by newlines) supported for the given target.

The crate type strings correspond to the values accepted by the `--crate-type` flag.

Intended to be used like this:

```bash
rustc --print=supported-crate-types -Zunstable-options --target=x86_64-unknown-linux-gnu
```

Example output for `x86_64-unknown-linux-gnu`:

```text
bin
cdylib
dylib
lib
proc-macro
rlib
staticlib
```
