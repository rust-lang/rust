# `staticlib-hide-internal-symbols`

When building a `staticlib`, this option hides all non-exported Rust-internal
symbols. On ELF targets, this sets `STV_HIDDEN` visibility. On Apple (Mach-O)
targets, this sets the `N_PEXT` (private external) bit.

This is a lightweight, zero-overhead operation: only the visibility/type byte of
each internal symbol is modified in-place.

Only symbols explicitly exported via `#[no_mangle]` or `#[export_name]` are left
unchanged. All other `GLOBAL`/`WEAK` symbols (including `pub(crate)` and `pub`
items without `#[no_mangle]`) are hidden.

This option can only be used with `--crate-type staticlib`. Using it with
other crate types will result in a compilation warning.

Supported on ELF targets (Linux, BSD, etc.) and Apple targets (macOS, iOS, etc.).
On unsupported targets (Windows), a warning is emitted and the flag has no effect.

This option can be combined with `-Zstaticlib-rename-internal-symbols`.
When both are enabled, symbols are both renamed and hidden.

## Comparison with `-Zdefault-visibility=hidden`

`-Zdefault-visibility=hidden` sets visibility at LLVM IR codegen time. It targets
shared objects (`cdylib`/`dylib`) and only affects the current crate's codegen.

`-Zstaticlib-hide-internal-symbols` patches visibility bytes post-compilation in
the final `.a` archive, which includes object files from all upstream static
dependencies. This means internal symbols from the entire dependency tree are
hidden, not just those from the current crate. Hidden symbols are also excluded
from the linker's global symbol table, which can slightly reduce final binary size.
