# `staticlib-rename-internal-symbols`

When building a `staticlib`, this option renames all non-exported Rust-internal
symbols by appending a `_rs{hash}` suffix and setting their ELF visibility to
`STV_HIDDEN`. This prevents symbol collisions when multiple Rust static
libraries are linked into the same final binary.

Only symbols explicitly exported via `#[no_mangle]` or `#[export_name]` are left
unchanged. All other `GLOBAL`/`WEAK` symbols (including `pub(crate)` and `pub`
items without `#[no_mangle]`) are renamed.

This option can only be used with `--crate-type staticlib`. Using it with
other crate types will result in a compilation error.

Currently only ELF targets are supported (Linux, BSD, etc.). On non-ELF
targets (macOS, Windows), a warning is emitted and the flag has no effect.
