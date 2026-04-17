# `staticlib-hide-internal-symbols`

When building a `staticlib`, this option hides all non-exported Rust-internal
symbols by setting their ELF visibility to `STV_HIDDEN`.

This is a lightweight, zero-overhead operation: only the visibility byte of each
internal symbol is modified in-place. No strtab manipulation or section header
copying is performed.

Only symbols explicitly exported via `#[no_mangle]` or `#[export_name]` are left
unchanged. All other `GLOBAL`/`WEAK` symbols (including `pub(crate)` and `pub`
items without `#[no_mangle]`) are hidden.

This option can only be used with `--crate-type staticlib`. Using it with
other crate types will result in a compilation error.

Currently only ELF targets are supported (Linux, BSD, etc.). On non-ELF
targets (macOS, Windows), a warning is emitted and the flag has no effect.

This option can be combined with `-Zstaticlib-rename-internal-symbols`.
When both are enabled, symbols are both renamed and hidden.
