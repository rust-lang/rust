# `wasm-c-abi`

This option controls whether Rust uses the spec-compliant C ABI when compiling
for the `wasm32-unknown-unknown` target.

This makes it possible to be ABI-compatible with all other spec-compliant Wasm targets
like `wasm32-wasip1`.

This compiler flag is perma-unstable, as it will be enabled by default in the
future with no option to fall back to the old non-spec-compliant ABI.
