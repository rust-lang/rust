# `staticlib-hide-internal-symbols`

When building a `staticlib`, this option hides all Rust-internal symbols
(except `rust_eh_personality`) by setting their ELF visibility to
`STV_HIDDEN`.

This option can only be used with `--crate-type staticlib`. Using it with
other crate types will result in a compilation error.

Currently only ELF targets are supported (Linux, BSD, etc.). On non-ELF
targets (macOS, Windows), a warning is emitted and the flag has no effect.
