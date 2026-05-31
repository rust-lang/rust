# `wasm-proc-macros`

This option controls whether to enable support for compiling and loading
`--crate-type=proc-macro` to/from WASM rather than the normal host dylib target.

Currently we expect that proc macros are compiled to the `wasm32-wasip2`
target. The exact target will likely change in the future. When this flag is
passed, both regular dylib proc macros and wasm proc macros are supported.
