# `staticlib-rename-internal-symbols`

When building a `staticlib`, this option renames all non-exported Rust-internal
symbols by appending a `_rs{hash}` suffix. This prevents symbol collisions when
multiple Rust static libraries are linked into the same final binary.

This option only renames symbols; it does not change their visibility.
Use `-Zstaticlib-hide-internal-symbols` in addition if you also want to hide
internal symbols.

Only symbols explicitly exported via `#[no_mangle]` or `#[export_name]` are left
unchanged. All other `GLOBAL`/`WEAK` symbols (including `pub(crate)` and `pub`
items without `#[no_mangle]`) are renamed.

This option can only be used with `--crate-type staticlib`. Using it with
other crate types will result in a compilation warning.

Supported on ELF targets (Linux, BSD, etc.) and Apple targets (macOS, iOS, etc.).
On unsupported targets (Windows), a warning is emitted and the flag has no effect.
