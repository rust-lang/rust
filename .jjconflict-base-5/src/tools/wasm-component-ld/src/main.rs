// See the `README.md` in this directory for what this tool is.

// The source for this crate lives at
// https://github.com/bytecodealliance/wasm-component-ld and the binary is
// independently used in other projects such as `wasi-sdk` so the `main`
// function is just reexported here to delegate. A Cargo dependency is used to
// facilitate version management in the Rust repository and work well with
// vendored/offline builds.
use wasm_component_ld::main;
